import streamlit as st
import time
import json
import random
import difflib
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# ==========================================
# 1. 配置与初始化 (Configuration & Init)
# ==========================================

st.set_page_config(
    page_title="PromptCraft - 提示词智能调试台",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Courier New', monospace; }
    .diff-added { background-color: #e6ffec; color: #24292e; padding: 2px; }
    .diff-removed { background-color: #ffebe9; color: #24292e; text-decoration: line-through; padding: 2px; }
    .metric-card { background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 4px solid #4e8cff; }
    .status-running { color: orange; font-weight: bold; }
    .status-idle { color: green; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 状态管理初始化
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = ""
if 'current_version' not in st.session_state:
    st.session_state.current_version = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'optimization_mode' not in st.session_state:
    st.session_state.optimization_mode = "LLM自动优化" # 或 "人工优化"

# ==========================================
# 2. 核心逻辑类 (Core Logic Classes)
# ==========================================

class MockLLMService:
    """
    模拟 LLM 服务 (支持执行、评价、优化)
    在实际生产中，这里替换为 OpenAI/Anthropic/Gemini 的 API 调用
    """
    @staticmethod
    def execute_prompt(prompt: str, variables: Dict[str, str], model_config: Dict) -> str:
        # 模拟 API 延迟
        time.sleep(0.5) 
        filled_prompt = prompt
        for k, v in variables.items():
            filled_prompt = filled_prompt.replace(f"{{{{{k}}}}}", v)
        
        return f"[模拟LLM回答] 基于提示词长度 {len(filled_prompt)} 的生成结果。\n核心内容：{filled_prompt[:20]}..."

    @staticmethod
    def evaluate_quality(output: str, expected: str, model_config: Dict) -> float:
        # 模拟评分 (0-10)
        base_score = random.uniform(6.0, 9.5)
        # 简单的模拟逻辑：如果输出包含预期关键词，分数更高
        if expected and expected in output:
            base_score += 0.5
        return min(10.0, round(base_score, 1))

    @staticmethod
    def optimize_prompt(current_prompt: str, feedback: str, model_config: Dict) -> str:
        # 模拟优化：在末尾添加一些修饰词
        modifiers = ["请更简洁一点。", "使用更专业的术语。", "请分点论述。", "注意语气要温和。"]
        chosen = random.choice(modifiers)
        return f"{current_prompt}\n\n[优化指令]: {chosen}"

class SessionManager:
    """管理调试会话和历史记录"""
    @staticmethod
    def save_history(history: List[Dict]):
        filename = f"debug_session_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        return filename

    @staticmethod
    def generate_diff(text1: str, text2: str) -> str:
        d = difflib.Differ()
        diff = list(d.compare(text1.splitlines(), text2.splitlines()))
        html_diff = ""
        for line in diff:
            if line.startswith('+ '):
                html_diff += f"<div class='diff-added'>{line[2:]}</div>"
            elif line.startswith('- '):
                html_diff += f"<div class='diff-removed'>{line[2:]}</div>"
            elif line.startswith('  '):
                html_diff += f"<div>{line[2:]}</div>"
        return html_diff

# ==========================================
# 3. 侧边栏配置 (Sidebar Configuration)
# ==========================================

with st.sidebar:
    st.header("⚙️ 调试控制台")
    
    st.subheader("1. 迭代设置")
    iter_mode = st.radio("迭代模式", ["自动迭代 (连续)", "交互式 (人工确认)"])
    max_iters = st.slider("最大迭代次数", 1, 10, 3)
    
    st.subheader("2. LLM 配置")
    with st.expander("API 参数设置"):
        exec_model = st.selectbox("执行模型", ["gpt-4o", "gemini-1.5-pro", "claude-3-5-sonnet"])
        eval_model = st.selectbox("评价模型", ["gpt-4o", "gpt-3.5-turbo"])
        temp = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    st.subheader("3. 策略配置")
    rollback_enabled = st.checkbox("启用质量回滚 (分数下降时)", value=True)
    rollback_threshold = st.number_input("回滚阈值 (分差)", 0.1, 2.0, 0.5)

    st.divider()
    if st.button("🗑️ 清空历史记录"):
        st.session_state.history = []
        st.session_state.current_version = 0
        st.rerun()

# ==========================================
# 4. 主界面 (Main Interface)
# ==========================================

st.title("🧩 Prompt Debugger Pro")

# -----------------
# Tab 1: 输入与设计
# -----------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 提示词设计 (Prompt)")
    st.caption("使用 {{variable}} 标记可变部分")
    
    # 默认模板
    default_prompt = """你是一个专业的翻译助手。
请将以下文本翻译成{{target_language}}：
内容：{{source_text}}

要求：
1. 保持原意
2. 风格优雅"""
    
    prompt_input = st.text_area(
        "Prompt Template", 
        value=st.session_state.get('current_prompt') or default_prompt, 
        height=300,
        key="prompt_editor"
    )
    st.session_state.current_prompt = prompt_input

with col2:
    st.subheader("🧪 测试用例 (Variables)")
    
    # 解析变量
    import re
    vars_found = re.findall(r"{{(.*?)}}", prompt_input)
    unique_vars = sorted(list(set(vars_found)))
    
    var_inputs = {}
    if unique_vars:
        for v in unique_vars:
            var_inputs[v] = st.text_input(f"变量: {v}", key=f"var_{v}")
    else:
        st.info("未检测到变量，将作为静态提示词运行。")
        
    st.subheader("🎯 预期标准 (Ground Truth)")
    expected_output = st.text_area("预期回答 (用于自动评分参考)", height=100)

# -----------------
# Tab 2: 调试与执行
# -----------------

st.divider()
st.subheader("🚀 调试执行")

control_col, status_col = st.columns([1, 3])

with control_col:
    start_btn = st.button("开始调试流程", type="primary", use_container_width=True)
    opt_path = st.selectbox("优化路径", ["LLM自动优化", "人工介入修改"])

with status_col:
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

# 结果展示容器
result_container = st.container()

# -----------------
# 逻辑执行 (Execution Logic)
# -----------------

if start_btn:
    st.session_state.is_running = True
    current_prompt_ver = prompt_input
    
    for i in range(max_iters):
        # 1. 状态更新
        status_placeholder.markdown(f"**⏳ 正在执行第 {i+1}/{max_iters} 轮迭代...**")
        progress_bar.progress((i + 1) / max_iters)
        
        # 2. 调用执行 LLM
        output = MockLLMService.execute_prompt(
            current_prompt_ver, 
            var_inputs, 
            {"model": exec_model, "temp": temp}
        )
        
        # 3. 调用评价 LLM
        score = MockLLMService.evaluate_quality(output, expected_output, {"model": eval_model})
        
        # 4. 记录数据
        record = {
            "version": st.session_state.current_version + 1,
            "timestamp": datetime.now().isoformat(),
            "prompt": current_prompt_ver,
            "output": output,
            "score": score,
            "diff_html": "", # 与上一版本对比
            "status": "Success"
        }
        
        # 5. 生成 Diff 与 回滚逻辑
        is_rollback = False
        if st.session_state.history:
            prev_record = st.session_state.history[-1]
            record["diff_html"] = SessionManager.generate_diff(prev_record["prompt"], current_prompt_ver)
            
            # 回滚检查
            if rollback_enabled and (prev_record["score"] - score > rollback_threshold):
                record["status"] = "Rollback Triggered"
                is_rollback = True
                status_placeholder.warning(f"⚠️ 检测到质量下降 (Score {score} < {prev_record['score']})，触发回滚。")
                current_prompt_ver = prev_record["prompt"] # 恢复旧提示词
        
        st.session_state.history.append(record)
        st.session_state.current_version += 1
        
        # 6. 展示当前结果
        with result_container:
            with st.expander(f"Iter #{i+1} - Score: {score} - {record['status']}", expanded=True):
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown("**生成结果:**")
                    st.info(output)
                with c2:
                    st.markdown("**Prompt 变更:**")
                    if record["diff_html"]:
                        st.markdown(record["diff_html"], unsafe_allow_html=True)
                    else:
                        st.caption("初始版本")
        
        # 7. 优化阶段 (准备下一轮)
        if i < max_iters - 1 and not is_rollback:
            if iter_mode == "交互式 (人工确认)":
                st.session_state.is_running = False
                status_placeholder.success(f"第 {i+1} 轮完成。等待人工确认...")
                break # 实际应用中这里需要更复杂的暂停逻辑，Streamlit 中通常通过 rerun 实现
            
            # 自动优化逻辑
            if opt_path == "LLM自动优化":
                current_prompt_ver = MockLLMService.optimize_prompt(current_prompt_ver, output, {})
                status_placeholder.info("🤖 AI 正在优化提示词...")
                time.sleep(1)
            else:
                # 如果是人工路径且自动模式，这里为了演示继续循环，实际应暂停
                pass
                
    st.session_state.is_running = False
    status_placeholder.success("✅ 调试流程结束")


# ==========================================
# 5. 历史与分析 (History & Analysis)
# ==========================================

if st.session_state.history:
    st.divider()
    st.header("📊 迭代分析报告")
    
    # 指标趋势图
    hist_df = pd.DataFrame(st.session_state.history)
    st.line_chart(hist_df, x="version", y="score")
    
    # 历史记录表格
    st.dataframe(
        hist_df[["version", "score", "status", "timestamp"]],
        use_container_width=True
    )
    
    # 导出功能
    col_export, col_dummy = st.columns([1, 4])
    with col_export:
        history_json = json.dumps(st.session_state.history, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 导出调试报告 (JSON)",
            data=history_json,
            file_name=f"prompt_debug_report_{int(time.time())}.json",
            mime="application/json"
        )

# ==========================================
# 6. 页脚
# ==========================================
st.markdown("---")
st.caption("Prompt Debugger Tool v1.0 | Local Mode | No Database Required")