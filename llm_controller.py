"""
llm_controller.py
================
LLM Controller：仅替换原 MARL 项目的 **action 生成层**（MAPPO 的 ``self.action(...)``），
环境（``CustomMultiAgentEnv``、``XuanCeMultiAgentEnvWrapper``、``DummyVecMultiAgentEnv``）**不做任何修改**。

格式契约（与 MAPPO 保持一致，见 main.py:2910-2916 与
docs/xuance-master/xuance/torch/agents/multi_agent_rl/mappo_agents.py:148-149）：

    输入 obs_dict  : list[dict]  （len = num_envs，此处 parallels=1）
                   每项 { 'agent_k1': np.ndarray[obs_len],
                          'agent_k2': np.ndarray[obs_len],
                          'agent_k3': np.ndarray[obs_len],
                          'agent_g' : np.ndarray[obs_len] }  取值 [0,1]

    输出 actions  : list[dict]  （len = num_envs）
                   每项 { 'agent_k1': np.ndarray[4], ..., 'agent_g': np.ndarray[4] }  取值 [-1,1]
                   （env.step 内部再经 np.clip((a+1)/2, 0, 1) 缩放到 [0,1] 作为资源调整比例）

    env.step     : envs.step(actions) → next_obs, reward, terminated, truncated, info

后端说明（零新依赖）：
    * 默认通过 ``requests`` 调用 **OpenAI 兼容** 的 /chat/completions 接口。
    * 默认指向部门本地化模型服务（《部门本地化模型使用说明V1.7》），base_url
      https://llm.example.invalid/v1，可用模型：qwen3-next-80b-a3b-instruct /
      qwen3-coder-next / deepseek-v4-flash-0731。
    * 必须在服务端 https://llm.example.invalid/ 注册并获取 token 填入密钥；
      未配置 API 时自动回退到启发式策略（``_heuristic_actions``），保证链路可跑通。
    * 配置读取顺序：config Namespace（如有）> 环境变量 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL。
"""

import json
import os
import re
import logging

import numpy as np

try:
    import requests
except ImportError:  # 无 requests 时降级（仅当完全未配置 API 时才可行）
    requests = None

DEFAULT_AGENT_KEYS = ["agent_k1", "agent_k2", "agent_k3", "agent_g"]
DEFAULT_ACTION_DIM = 4  # main.py:232-240
DEFAULT_MAX_LINKS = 36  # main.py:214, max(re_n_set_num_links)
OBS_UNKNOWN_SEGMENTS = False  # 占位常量，仅用于可读性


class LLMController(object):
    """
    替代 MAPPO action 生成的 Controller。

    Parameters
    ----------
    config : argparse.Namespace, optional
        若提供，优先从 config.llm_base_url / config.llm_api_key / config.llm_model /
        config.agent_keys / config.action_dim / config.max_num_links 读取；缺失字段回退到环境变量。
    agent_keys : list[str], optional
        智能体名列表，默认 ["agent_k1","agent_k2","agent_k3","agent_g"]（main.py:211）。
    action_dim : int, optional
        每个智能体的动作维度，默认 4（main.py:232-240）。
    max_num_links : int, optional
        补零后链路观测上限 max_num_links，用于把 obs 解析成 node/link/adj 分段，默认 36。
    temperature : float, optional
        LLM 采样温度。
    system_prompt : str, optional
        覆盖默认的系统提示词。
    use_llm : bool, optional
        True 强制走 LLM；False 强制 heuristic；None(默认) 自动：config/环境变量齐备则走 LLM。
    """

    def __init__(self,
                 config=None,
                 agent_keys=None,
                 action_dim=DEFAULT_ACTION_DIM,
                 max_num_links=DEFAULT_MAX_LINKS,
                 temperature=0.2,
                 system_prompt=None,
                 use_llm=None):
        self.agent_keys = list(agent_keys) if agent_keys else list(DEFAULT_AGENT_KEYS)
        self.action_dim = int(action_dim)
        self.max_num_links = int(max_num_links)
        self.temperature = temperature
        self.config = config

        # --- 后端配置：config > 环境变量 > 默认（部门本地化模型服务） ---
        self.base_url = self._get_cfg("llm_base_url", "LLM_BASE_URL", "https://llm.example.invalid/v1")
        self.api_key = self._get_cfg("llm_api_key", "LLM_API_KEY", None)
        self.model = self._get_cfg("llm_model", "LLM_MODEL", "deepseek-v4-flash-0731")
        self.timeout = float(self._get_cfg("llm_timeout", "LLM_TIMEOUT", "60"))

        if use_llm is None:
            self.use_llm = bool(self.api_key)
        else:
            self.use_llm = bool(use_llm)

        if self.use_llm and requests is None:
            logging.warning("[LLMController] requests 不可用且未安装，回退到启发式策略。")
            self.use_llm = False

        self.system_prompt = system_prompt if system_prompt else self._default_system_prompt()

        # 最近一次输出（便于调试 / 记录日志）
        self.last_actions = None
        self.last_text = None

        mode = "LLM -> %s @ %s" % (self.model, self.base_url) if self.use_llm else "heuristic (fallback)"
        logging.info("[LLMController] action generation backend: %s" % mode)

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def get_actions(self, obs_dict, info=None):
        """
        由观测生成动作（替代 MAPPO 的 self.action(...) 输出 action 的部分）。

        Parameters
        ----------
        obs_dict : dict 或 list[dict]
            单个 dict 或 list（len=num_envs），每个 dict 为 agent -> np.ndarray。
        info : list[dict], optional
            环境 info（当前未使用，保留为参数以匹配循环签名）。

        Returns
        -------
        actions : list[dict]
            [{agent: np.ndarray[action_dim]}]，取值已被 clip 到 [-1, 1]，可直喂 envs.step(actions)。
        """
        obs_list = self._normalize_obs_list(obs_dict)
        actions = []
        for obs in obs_list:
            if self.use_llm:
                parsed = self._llm_actions(obs)
            else:
                parsed = self._heuristic_actions(obs)
            actions.append({
                k: np.clip(np.asarray(parsed.get(k, self._neutral_action()), dtype=np.float32),
                           -1.0, 1.0).reshape(self.action_dim)
                for k in self.agent_keys
            })
        self.last_actions = actions
        return actions

    def act_and_step(self, envs, obs_dict, info=None):
        """
        便捷方法：生成动作并直接推进环境（等价于原 self.action(...) + self.envs.step(...)）。
        仅用于测试/推理循环；不修改环境。
        """
        actions = self.get_actions(obs_dict, info)
        return envs.step(actions)

    # ------------------------------------------------------------------ #
    # obs -> prompt -> LLM -> JSON -> action
    # ------------------------------------------------------------------ #
    def _llm_actions(self, obs):
        """调用 LLM 生成一个环境的所有 agent 动作；失败则回退启发式。"""
        try:
            user_msg = self._build_prompt(obs)
            text = self._call_chat(user_msg)
            self.last_text = text
            return self._parse_actions_json(text)
        except Exception as e:  # 解析失败/网络失败 -> 降级，不崩训练
            logging.warning("[LLMController] LLM 调用/解析失败 (%s)，回退启发式。" % repr(e))
            return self._heuristic_actions(obs)

    def _build_prompt(self, obs):
        """把单个环境的多智能体 obs 组装成给 LLM 的用户消息。"""
        lines = []
        summary = self._summarize_obs(obs)
        for k in self.agent_keys:
            seg = summary.get(k, {})
            lines.append(
                "- %s: 节点段均值=%.4f(共%d), 链路段均值=%.4f(共%d), 邻接段均值=%.4f(共%d), 整体均值=%.4f"
                % (k,
                   seg.get("node_mean", 0.0), seg.get("node_len", 0),
                   seg.get("link_mean", 0.0), seg.get("link_len", 0),
                   seg.get("adj_mean", 0.0), seg.get("adj_len", 0),
                   seg.get("overall_mean", 0.0)))
        # 补充逐 agent 的整体统计，帮助 LLM 理解资源紧张程度
        detail = ["%s: overall_mean=%.4f overall_std=%.4f max=%.4f nnz_frac=%.3f"
                 % (k, s.get("overall_mean", 0.0), s.get("overall_std", 0.0),
                    s.get("overall_max", 0.0), s.get("nnz_frac", 0.0))
                 for k, s in summary.items()]
        lines.append("观测统计明细：\n" + "\n".join(detail))
        lines.append("请依据上述观测，返回 4 个智能体的动作（见约束）。仅输出 JSON，不要输出其他内容。")
        return "\n".join(lines)

    def _call_chat(self, user_msg):
        """OpenAI 兼容 chat/completions 调用。"""
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_json(text):
        """从模型输出中提取 JSON 对象（容忍 ```json 代码块包裹 / 前后杂文）。"""
        if not text:
            raise ValueError("empty LLM response")
        text = text.strip()
        # 去掉 ```json ... ``` 代码块围栏
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if fence:
            text = fence.group(1)
        # 直接找最外层花括号
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found in response")
        return json.loads(text[start:end + 1])

    def _parse_actions_json(self, text):
        parsed = self._extract_json(text)
        if not isinstance(parsed, dict):
            raise ValueError("parsed JSON is not a dict")
        out = {}
        for k in self.agent_keys:
            v = parsed.get(k)
            arr = np.asarray(v, dtype=np.float32).reshape(self.action_dim)
            if arr.size != self.action_dim:
                raise ValueError("agent %s action dim=%d != %d" % (k, arr.size, self.action_dim))
            out[k] = np.clip(arr, -1.0, 1.0)
        return out

    # ------------------------------------------------------------------ #
    # 启发式回退（未配置 LLM 或调用失败时）
    # ------------------------------------------------------------------ #
    def _heuristic_actions(self, obs):
        """默认启发式：动作全 0（env.step 内映射到 0.5，表示资源调整比例取中间值）。"""
        actions = {}
        for k in self.agent_keys:
            arr = obs.get(k)
            if arr is not None:
                # 简单可解释策略：观测均值越低（资源越空闲）越少让出/占用 -> 动作取 (0.5 - mean) 的缩放
                mean = float(np.clip(np.mean(arr), 0.0, 1.0))
                v = float(np.clip(0.5 - mean, -1.0, 1.0))
                actions[k] = np.full(self.action_dim, v, dtype=np.float32)
            else:
                actions[k] = np.zeros(self.action_dim, dtype=np.float32)
        return actions

    def _neutral_action(self):
        return np.zeros(self.action_dim, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # obs 分段解析
    # ------------------------------------------------------------------ #
    def _summarize_obs(self, obs):
        """把单个环境各 agent 的 obs 解析为 node/link/adj 三段统计摘要。

        obs 布局（main.py:2164-2248 / 2250-2339）：
            node 段 : 每节点 5 个特征，长度 = n_node * 5
            link 段 : 每链路 3 个特征，补零到 max_num_links * 3
            adj 段  : 每条链路两端点 2 个值，补零到 max_num_links * 2
         其中 n_node = len(obs) // 5 - max_num_links（当 len(obs)%5==0 且结论为正）。
        """
        out = {}
        for k in self.agent_keys:
            arr = obs.get(k)
            if arr is None:
                out[k] = {}
                continue
            arr = np.asarray(arr, dtype=np.float32)
            n_total = arr.size
            s = {
                "overall_mean": float(np.mean(arr)),
                "overall_std": float(np.std(arr)),
                "overall_max": float(np.max(arr)),
                "nnz_frac": float(np.count_nonzero(arr)) / max(n_total, 1),
            }
            node_len, link_len, adj_len = 0, 0, 0
            if n_total > 0 and n_total % 5 == 0:
                node_len = n_total // 5 - self.max_num_links
                if node_len > 0:
                    link_len = self.max_num_links * 3
                    adj_len = self.max_num_links * 2
            if node_len > 0:
                s.update({
                    "node_len": node_len,
                    "node_mean": float(np.mean(arr[:node_len * 5])),
                    "link_len": link_len,
                    "link_mean": float(np.mean(arr[node_len * 5: node_len * 5 + link_len])),
                    "adj_len": adj_len,
                    "adj_mean": float(np.mean(arr[node_len * 5 + link_len: node_len * 5 + link_len + adj_len])),
                })
            out[k] = s
        return out

    @staticmethod
    def _normalize_obs_list(obs_dict):
        """接受 dict 或 list[dict]，统一返回 list[dict]。"""
        if isinstance(obs_dict, (list, tuple)):
            return list(obs_dict)
        return [obs_dict]

    def _get_cfg(self, attr, env_name, default):
        if self.config is not None and hasattr(self.config, attr):
            v = getattr(self.config, attr)
            if v is not None:
                return v
        return os.environ.get(env_name, default)

    # ------------------------------------------------------------------ #
    # 默认系统提示词
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_system_prompt():
        return (
            "你是一个多租户虚拟网络资源分配控制器。系统中有 4 个智能体协同工作："
            "agent_k1、agent_k2、agent_k3 代表已有虚拟业务网络（需要向新网络让出节点计算/缓存与链路带宽资源），"
            "agent_g 代表新到达的虚拟业务网络（需要占用物理网络的节点与链路资源）。\n"
            "每个智能体每步输出一个 4 维连续动作 a（每个分量取值 [-1,1]）。"
            "该动作会被环境映射到 [0,1]，作为对应节点/链路资源的调整比例。"
            "你应依据观测中的资源容量与利用率，在\"保障已有业务\"与\"容纳新业务\"之间给出协调策略，"
            "避免资源利用接近饱和时继续激进让出，避免资源空闲时不敢让出。\n"
            "请严格输出一个 JSON 对象，形如：\n"
            '{"agent_k1": [0.1, -0.2, 0.3, -0.1], "agent_k2": [0.0, 0.0, 0.0, 0.0], '
            '"agent_k3": [0.2, 0.1, -0.1, 0.2], "agent_g": [0.3, 0.3, 0.2, 0.2]}'
        )


# ------------------------------------------------------------------ #
# 自测：不依赖真实环境，仅校验格式契约
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def _dummy_obs(length=240, seed=0):
        rng = np.random.RandomState(seed)
        o = {}
        for k in ["agent_k1", "agent_k2", "agent_k3", "agent_g"]:
            o[k] = rng.rand(length).astype(np.float32)
        return o

    ctrl = LLMController(use_llm=False)  # 强制启发式，格式自检
    obs_dict = _dummy_obs()
    acts = ctrl.get_actions(obs_dict)
    assert len(acts) == 1
    for k in ["agent_k1", "agent_k2", "agent_k3", "agent_g"]:
        a = acts[0][k]
        assert a.shape == (4,), a.shape
        assert a.dtype == np.float32
        assert a.min() >= -1.0 and a.max() <= 1.0
    print("llm_controller format self-test OK")
    print("sample action:", acts[0])
    print("summary keys:", ctrl._summarize_obs(obs_dict)["agent_k1"])
