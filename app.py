import streamlit as st
import json
import os
import time
import datetime
import uuid
import copy
from jinja2 import Template
import difflib
import random

# --- 配置与常量 ---
SESSION_DIR = "prompt_sessions"
PRESET_DIR = "prompt_presets"
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(PRESET_DIR, exist_ok=True)

st.set_page_config(page_title="PromptCraft - 提示词调试工作台", layout="wide", page_icon="🔧")

# --- 核心逻辑类 ---

class DataManager:
    """数据持久化层：处理文件存储"""
    @staticmethod
    def save_session(session_data, session_name=None):
        if not session_name:
            session_name = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        path = os.path.join(SESSION_DIR, f"{session_name}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def list_sessions():
        files = [f for f in os.listdir(SESSION_DIR) if f.endswith('.json')]
        return sorted(files, reverse=True)

class LLMEngine:
    """LLM调用代理服务"""
    def __init__(self, api_key, base_url="https://api.openai.com/v1"):
        self.api_key = api_key
        # 这里预留实际的OpenAI调用接口
        # 实际项目中应使用 openai.OpenAI client
        
    def execute(self, prompt, model_config, mock=False):
        """执行LLM生成回答"""
        if mock:
            time.sleep(1) # 模拟延迟
            return f"这是基于提示词产生的模拟回答。\n\n提示词重点：{prompt[:20]}...\n设定参数：{model_config['temperature']}"
        # 实现真实的API调用...
        return "请在设置中关闭模拟模式以连接真实API"

    def evaluate(self, question, actual_output, expected_output, model_config, mock=False):
        """评价LLM：打分并给出理由"""
        if mock:
            time.sleep(1)
            score = random.randint(60, 95)
            return {
                "score": score,
                "reason": f"模拟评分：{score}。回答结构清晰，但细节略有差异。与预期结果的匹配度尚可。"
            }
        # 实现真实的评价Prompt...
        return {"score": 0, "reason": "未连接API"}

    def optimize(self, current_prompt, evaluation_result, model_config, mock=False):
        """优化LLM：生成新的提示词建议"""
        if mock:
            time.sleep(1)
            return f"{current_prompt}\n\n[优化建议]：增加具体的输出格式限制，并强调语气更加专业。"
        # 实现真实的优化Prompt...
        return current_prompt

# --- 辅助函数 ---

def render_prompt(template_str, variables_json):
    try:
        if not variables_json.strip():
            return template_str
        vars_dict = json.loads(variables_json)
        template = Template(template_str)
        return template.render(**vars_dict)
    except Exception as e:
        return f"Error rendering template: {str(e)}"

def get_diff_html(text1, text2):
    """生成HTML格式的文本差异对比"""
    d = difflib.Differ()
    diff = list(d.compare(text1.splitlines(), text2.splitlines()))
    html = []
    for line in diff:
        if line.startswith('+ '):
            html.append(f'<div style="background-color: #e6ffec; color: #155724;">{line}</div>')
        elif line.startswith('- '):
            html.append(f'<div style="background-color: #ffe6e6; color: #721c24;">{line}</div>')
        elif line.startswith('? '):
            continue
        else:
            html.append(f'<div style="color: #666;">{line}</div>')
    return "".join(html)

# --- 界面状态管理 ---

if 'history' not in st.session_state:
    st.session_state.history = [] # 存储每次迭代的详细记录
if 'current_version' not in st.session_state:
    st.session_state.current_version = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'logs' not in st.session_state:
    st.session_state.logs = []

def log(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")

# --- 侧边栏：配置管理 ---

with st.sidebar:
    st.header("⚙️ 全局配置")
    
    # 模拟模式开关
    use_mock = st.toggle("启用模拟模式 (无需API Key)", value=True)
    
    st.divider()
    
    st.subheader("🤖 模型配置")
    
    with st.expander("1. 执行模型 (Executor)", expanded=True):
        exec_model = st.selectbox("Model", ["gpt-4o", "gpt-3.5-turbo", "claude-3-5-sonnet"], key="exec_m")
        exec_temp = st.slider("Temperature", 0.0, 1.0, 0.7, key="exec_t")
    
    with st.expander("2. 评价模型 (Evaluator)"):
        eval_model = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo"], key="eval_m")
        st.caption("负责为执行结果打分 (0-100)")

    with st.expander("3. 优化模型 (Optimizer)"):
        opt_model = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo"], key="opt_m")
        st.caption("负责根据评价结果修改提示词")

    st.divider()
    st.subheader("💾 模板管理")
    preset_name = st.text_input("保存当前配置为模板")
    if st.button("保存模板"):
        st.success(f"模板 {preset_name} 已保存 (模拟)")

# --- 主界面 ---

st.title("🚀 提示词智能调试工作台")

# 1. 核心输入区域
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📝 提示词设计 (Prompt Template)")
    prompt_input = st.text_area(
        "支持 {{variable}} 语法", 
        height=300, 
        value="你是一个专业的翻译助手。\n请将以下内容翻译成中文：\n\n{{source_text}}",
        key="prompt_input_area"
    )

with col2:
    st.subheader("🧩 变量与预期")
    variables_input = st.text_area("变量 (JSON格式)", value='{\n  "source_text": "To be or not to be, that is the question."\n}', height=120)
    expected_output = st.text_area("预期回答 (用于自动评分)", value="生存还是毁灭，这是一个问题。", height=120)
    
    st.info("💡 提示：在左侧使用 {{variable}} 标记可变部分，在右侧JSON中定义具体值。")

# 2. 控制台与迭代设置
st.divider()

ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

with ctrl_col1:
    iteration_mode = st.radio("迭代模式", ["自动迭代", "交互式迭代"], horizontal=True)

with ctrl_col2:
    max_iterations = st.number_input("最大迭代次数", min_value=1, max_value=10, value=3)

with ctrl_col3:
    auto_rollback = st.checkbox("启用智能回滚", value=True, help="当新版本得分低于旧版本时，自动恢复")

with ctrl_col4:
    start_btn = st.button("▶️ 开始调试", type="primary", use_container_width=True)
    reset_btn = st.button("🔄 重置会话", use_container_width=True)

if reset_btn:
    st.session_state.history = []
    st.session_state.logs = []
    st.session_state.current_version = 0
    st.rerun()

# --- 核心处理逻辑 ---

llm_engine = LLMEngine(api_key="mock" if use_mock else os.getenv("OPENAI_API_KEY"))

if start_btn:
    st.session_state.is_running = True
    current_prompt = prompt_input
    
    # 初始化/继续迭代循环
    progress_bar = st.progress(0)
    
    for i in range(max_iterations):
        iter_num = i + 1
        log(f"--- 开始迭代 #{iter_num} ---")
        
        # 1. 渲染变量
        final_prompt = render_prompt(current_prompt, variables_input)
        
        # 2. 执行调用
        log("正在调用执行模型...")
        actual_output = llm_engine.execute(
            final_prompt, 
            {"model": exec_model, "temperature": exec_temp}, 
            mock=use_mock
        )
        
        # 3. 质量评估
        log("正在评估结果质量...")
        eval_result = llm_engine.evaluate(
            final_prompt, actual_output, expected_output,
            {"model": eval_model}, mock=use_mock
        )
        score = eval_result['score']
        log(f"当前得分: {score}/100")
        
        # 记录本次迭代数据
        iteration_data = {
            "version": iter_num,
            "prompt_template": current_prompt,
            "final_prompt": final_prompt,
            "output": actual_output,
            "evaluation": eval_result,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # 4. 智能回滚逻辑
        rollback_triggered = False
        if auto_rollback and len(st.session_state.history) > 0:
            last_score = st.session_state.history[-1]['evaluation']['score']
            if score < last_score:
                log(f"⚠️ 警告: 质量下降 ({score} < {last_score})，触发回滚。")
                rollback_triggered = True
                # 回滚提示词到上一个版本，但记录这次失败的尝试
                iteration_data['status'] = 'rolled_back'
                current_prompt = st.session_state.history[-1]['prompt_template']
            else:
                iteration_data['status'] = 'accepted'
        else:
            iteration_data['status'] = 'accepted'
            
        st.session_state.history.append(iteration_data)
        progress_bar.progress((i + 1) / max_iterations)
        
        # 5. 优化逻辑 (如果不是最后一次)
        if i < max_iterations - 1:
            if iteration_mode == "交互式迭代":
                st.session_state.awaiting_user = True
                # 在真实应用中，这里需要一种机制打断循环等待用户输入
                # Streamlit的机制很难在循环中直接暂停，通常通过rerun处理
                # 此处简化处理：交互模式下每次只跑一轮，需要用户再次点击继续
                log("交互模式：等待用户确认下一步...")
                break 
            
            if not rollback_triggered:
                log("正在生成优化建议...")
                current_prompt = llm_engine.optimize(
                    current_prompt, eval_result, 
                    {"model": opt_model}, mock=use_mock
                )
        
    st.session_state.is_running = False
    st.success("调试流程完成！")

# --- 结果可视化区域 ---

st.divider()
st.subheader("📊 调试报告")

if not st.session_state.history:
    st.info("暂无调试数据，请点击开始调试。")
else:
    # 历史记录选项卡
    tab1, tab2, tab3, tab4 = st.tabs(["🔎 结果详情", "📈 迭代历史趋势", "↔️ 提示词差异对比", "📟 运行日志"])
    
    with tab1:
        # 选择要查看的版本
        versions = [f"v{h['version']} (Score: {h['evaluation']['score']})" for h in st.session_state.history]
        selected_v_idx = st.selectbox("选择版本查看详情", range(len(versions)), format_func=lambda x: versions[x])
        
        sel_data = st.session_state.history[selected_v_idx]
        
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("#### 提示词 (Prompt)")
            st.code(sel_data['prompt_template'], language="markdown")
            st.markdown(f"**状态:** `{sel_data['status']}`")
            
        with r_col2:
            st.markdown("#### 输出结果 (Output)")
            st.info(sel_data['output'])
            st.markdown("#### 评价 (Evaluation)")
            st.metric("质量评分", sel_data['evaluation']['score'])
            st.warning(f"评价理由: {sel_data['evaluation']['reason']}")

            # 人工干预区
            st.markdown("---")
            st.markdown("**人工优化路径**")
            new_manual_prompt = st.text_area("基于此版本手动修改", value=sel_data['prompt_template'], key="manual_edit")
            if st.button("采纳此手动修改并更新输入框"):
                st.session_state.prompt_input_area = new_manual_prompt # 注意：这需要回调技巧或rerun才能生效
                st.info("提示词已更新到上方输入框")

    with tab2:
        # 使用简单的图表显示分数趋势
        scores = [h['evaluation']['score'] for h in st.session_state.history]
        st.line_chart(scores)
        st.dataframe(st.session_state.history)

    with tab3:
        if len(st.session_state.history) > 1:
            v_a_idx = st.selectbox("版本 A", range(len(versions)), index=0, key="diff_a")
            v_b_idx = st.selectbox("版本 B", range(len(versions)), index=len(versions)-1, key="diff_b")
            
            prompt_a = st.session_state.history[v_a_idx]['prompt_template']
            prompt_b = st.session_state.history[v_b_idx]['prompt_template']
            
            st.markdown("### 差异视图 (Version A vs Version B)")
            diff_html = get_diff_html(prompt_a, prompt_b)
            st.markdown(diff_html, unsafe_allow_html=True)
        else:
            st.write("需要至少两次迭代才能进行对比。")

    with tab4:
        for log_msg in st.session_state.logs:
            st.text(log_msg)

    # 导出功能
    st.divider()
    report_json = json.dumps(st.session_state.history, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 导出完整调试报告 (JSON)",
        data=report_json,
        file_name="prompt_debug_report.json",
        mime="application/json"
    )