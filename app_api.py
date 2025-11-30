import streamlit as st
import os
import json
import time
import datetime
import difflib
import re
from jinja2 import Template
from openai import OpenAI

# ==========================================
# 1. 配置与初始化
# ==========================================

# 页面配置
st.set_page_config(
    page_title="PromptCraft - 智能提示词调试工作台",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 目录初始化
SESSION_DIR = "prompt_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# Session State 初始化
if 'history' not in st.session_state:
    st.session_state.history = []
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = ""
if 'iteration_count' not in st.session_state:
    st.session_state.iteration_count = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# ==========================================
# 2. 核心逻辑类 (后端服务)
# ==========================================

class DataManager:
    """数据持久化层：不依赖数据库，使用JSON文件"""
    @staticmethod
    def save_session(session_data, prefix="session"):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{timestamp}.json"
        path = os.path.join(SESSION_DIR, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            return path
        except Exception as e:
            st.error(f"保存失败: {str(e)}")
            return None

class LLMEngine:
    """LLM 调用引擎：集成 OpenAI SDK"""
    def __init__(self, api_key, base_url=None):
        self.client = None
        self.mock = False
        if not api_key:
            self.mock = True
        else:
            try:
                self.client = OpenAI(
                    api_key=api_key,
                    # base_url=base_url if base_url and base_url.strip() else "https://api.openai.com/v1"
                    base_url="https://api-inference.modelscope.cn/v1"
                )
            except Exception as e:
                st.error(f"API 初始化失败: {e}")
                self.mock = True

    def _clean_json(self, text):
        """清洗并提取 JSON"""
        try:
            return json.loads(text)
        except:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                try: return json.loads(match.group(1))
                except: pass
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                try: return json.loads(match.group(1))
                except: pass
            return None

    def execute(self, prompt, config):
        """执行 Prompt"""
        if self.mock:
            time.sleep(1)
            return f"【模拟结果】根据 Prompt: {prompt[:30]}... 生成的回答。\n设置温度: {config.get('temperature')}"
        
        try:
            response = self.client.chat.completions.create(
                model=config.get("model", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 1000),
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Execute Error: {str(e)}"

    def evaluate(self, prompt, output, expected, config):
        """评估结果质量"""
        if self.mock:
            time.sleep(0.5)
            import random
            score = random.randint(60, 95)
            return {"score": score, "reason": "模拟评分模式：结果结构尚可，但细节需优化。"}

        eval_prompt = f"""
        你是一名严格的评判员。请根据预期标准评估实际输出。
        
        【Prompt】: {prompt}
        【实际输出】: {output}
        【预期输出/标准】: {expected}
        
        请输出 JSON 格式：
        {{
            "score": <0-100的整数>,
            "reason": "<简短评语>"
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model=config.get("model", "gpt-4"),
                messages=[{"role": "user", "content": eval_prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                stream=False,
            )
            res = self._clean_json(response.choices[0].message.content)
            return res if res else {"score": 0, "reason": "评分格式解析失败"}
        except Exception as e:
            return {"score": 0, "reason": f"Evaluate Error: {str(e)}"}

    def optimize(self, current_prompt, eval_result, config):
        """基于评估优化 Prompt"""
        if self.mock:
            time.sleep(1)
            return current_prompt + "\n\n(已基于评估结果自动优化：增加了具体的格式限制)"

        opt_prompt = f"""
        你是一名 Prompt 工程师。请根据评分优化提示词。
        
        【原提示词】: {current_prompt}
        【评分】: {eval_result.get('score')}
        【问题】: {eval_result.get('reason')}
        
        要求：
        1. 保持原有的 {{{{variable}}}} 占位符不变。
        2. 使用思维链或少样本技巧增强效果。
        3. 直接输出优化后的提示词内容，无需解释。
        """
        try:
            response = self.client.chat.completions.create(
                model=config.get("model", "gpt-4"),
                messages=[{"role": "user", "content": opt_prompt}],
                temperature=0.7,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Optimize Error: {str(e)}"

# ==========================================
# 3. 辅助函数
# ==========================================

def log(msg):
    t = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{t}] {msg}")

def render_prompt(template, variables_str):
    if not variables_str.strip():
        return template
    try:
        vars_dict = json.loads(variables_str)
        return Template(template).render(**vars_dict)
    except Exception as e:
        log(f"渲染错误: {e}")
        return template

def get_diff_html(text1, text2):
    d = difflib.Differ()
    diff = list(d.compare(text1.splitlines(), text2.splitlines()))
    html = []
    for line in diff:
        if line.startswith('+ '):
            html.append(f'<div style="background:#e6ffec;color:#155724;padding:2px;">{line}</div>')
        elif line.startswith('- '):
            html.append(f'<div style="background:#ffe6e6;color:#721c24;padding:2px;">{line}</div>')
        elif line.startswith('? '): continue
        else: html.append(f'<div style="color:#666;padding:2px;">{line}</div>')
    return "".join(html)

# ==========================================
# 4. 界面构建
# ==========================================

# --- 侧边栏：配置 ---
with st.sidebar:
    st.title("⚙️ 设置面板")
    
    st.subheader("1. API 配置")
    api_key = st.text_input("OpenAI API Key", placeholder="ms-06251eb2-c784-4b06-9009-9f7f2bd61602", type="password", help="留空则使用模拟模式")
    base_url = st.text_input("Base URL (可选)", placeholder="https://api-inference.modelscope.cn/v1")
    
    st.subheader("2. 模型路由")
    c1, c2 = st.columns(2)
    exec_model = c1.selectbox("执行模型", ["gpt-3.5-turbo", "gpt-4o", "gpt-4", 'Qwen/Qwen3-8B', 'deepseek-ai/DeepSeek-V3.1'], index=0)
    exec_temp = c2.slider("执行温度", 0.0, 1.0, 0.7)
    
    eval_model = st.selectbox("评价模型", ["gpt-4", "gpt-4o", 'Qwen/Qwen3-8B', 'deepseek-ai/DeepSeek-V3.1'], index=0, help="建议使用强模型进行评分")
    opt_model = st.selectbox("优化模型", ["gpt-4", "gpt-4o", 'Qwen/Qwen3-8B', 'deepseek-ai/DeepSeek-V3.1'], index=0, help="建议使用强模型进行重写")
    
    st.divider()
    st.info(f"模式: {'🔵 模拟模式' if not api_key else '🟢 API 模式'}")
    
    if st.button("🗑️ 清空历史"):
        st.session_state.history = []
        st.session_state.logs = []
        st.session_state.iteration_count = 0
        st.session_state.current_prompt = ""
        st.rerun()

# --- 主界面 ---
st.title("🛠️ PromptCraft 调试工具")

# 输入区域
col_input, col_config = st.columns([3, 2])

with col_input:
    st.subheader("提示词模板 (可变部分用 {{var}})")
    # 如果还没有当前Prompt，使用默认值
    default_prompt = "你是一个翻译助手。\n请将以下文本翻译成中文：\n{{text}}"
    if not st.session_state.current_prompt:
        st.session_state.current_prompt = default_prompt
        
    prompt_input = st.text_area("Prompt Input", value=st.session_state.current_prompt, height=250, key="ui_prompt_input")
    # 同步回 session_state (允许人工手动修改)
    st.session_state.current_prompt = prompt_input

with col_config:
    st.subheader("测试数据 & 预期")
    vars_input = st.text_area("变量 (JSON)", value='{\n  "text": "The quick brown fox jumps over the lazy dog."\n}', height=100)
    expect_input = st.text_area("预期回答 (用于评分)", value="这只敏捷的棕色狐狸跳过了这只懒惰的狗。", height=100)
    
    st.markdown("---")
    c_iter, c_mode = st.columns(2)
    max_iters = c_iter.number_input("迭代次数", 1, 10, 3)
    mode = c_mode.radio("迭代模式", ["自动连续", "交互式(单步)"], horizontal=True)
    
    auto_rollback = st.checkbox("📉 启用自动回滚 (分数下降时恢复)", value=True)

# 操作栏
st.divider()
b1, b2, b3 = st.columns([1, 1, 4])
start_btn = b1.button("▶️ 开始调试", type="primary", use_container_width=True)
stop_btn = b2.button("⏹️ 停止", use_container_width=True)

# ==========================================
# 5. 执行逻辑
# ==========================================

engine = LLMEngine(api_key, base_url)

if start_btn:
    st.session_state.is_running = True
    st.session_state.iteration_count = 0
    # 清空之前的日志，保留历史记录可选，这里选择清空本次会话的临时日志
    st.session_state.logs = [] 

if st.session_state.is_running:
    status_container = st.status("正在执行调试流程...", expanded=True)
    progress_bar = st.progress(0)
    
    # 确定循环次数：如果是交互式，实际上只跑 1 轮，但为了代码复用，我们在循环内控制 break
    target_loops = max_iters if mode == "自动连续" else 1
    
    current_p = st.session_state.current_prompt
    
    for i in range(target_loops):
        idx = st.session_state.iteration_count + 1
        
        # 如果达到最大次数，停止
        if idx > max_iters:
            st.success("达到最大迭代次数。")
            st.session_state.is_running = False
            break

        status_container.write(f"🔄 正在执行第 {idx} 轮迭代...")
        log(f"=== 开始第 {idx} 轮 ===")
        
        # 1. 渲染
        rendered_prompt = render_prompt(current_p, vars_input)
        
        # 2. 执行
        log("调用 LLM 执行...")
        output = engine.execute(rendered_prompt, {"model": exec_model, "temperature": exec_temp})
        
        # 3. 评价
        log("调用 LLM 评分...")
        eval_res = engine.evaluate(rendered_prompt, output, expect_input, {"model": eval_model})
        score = eval_res.get('score', 0)
        log(f"本轮得分: {score}")
        
        # 记录数据
        record = {
            "version": idx,
            "prompt_template": current_p,
            "rendered_prompt": rendered_prompt,
            "output": output,
            "evaluation": eval_res,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "normal"
        }
        
        # 4. 回滚判断
        rollback_triggered = False
        if auto_rollback and len(st.session_state.history) > 0:
            last_score = st.session_state.history[-1]['evaluation']['score']
            if score < last_score:
                log(f"⚠️ 分数下降 ({score} < {last_score})，触发回滚。")
                record['status'] = "rolled_back"
                rollback_triggered = True
                # 回滚提示词：重置为上一个有效版本
                current_p = st.session_state.history[-1]['prompt_template']
            else:
                record['status'] = "accepted"
        else:
            record['status'] = "accepted"
            
        st.session_state.history.append(record)
        st.session_state.iteration_count += 1
        progress_bar.progress(idx / max_iters)
        
        # 5. 优化 (如果是最后一轮则不优化，或者触发回滚后不基于当前坏结果优化)
        if idx < max_iters:
            if mode == "交互式(单步)":
                log("交互模式：暂停等待用户确认...")
                st.session_state.is_running = False # 停止运行，等待用户手动再次点击
                st.info(f"第 {idx} 轮完成。请查看结果，如果需要优化下一版，请再次点击开始。")
                break
            
            if not rollback_triggered:
                log("生成优化建议中...")
                optimized_prompt = engine.optimize(current_p, eval_res, {"model": opt_model})
                current_p = optimized_prompt
                st.session_state.current_prompt = optimized_prompt # 更新全局状态
        else:
            st.session_state.is_running = False
            st.success("所有迭代已完成！")

    status_container.update(label="流程结束", state="complete", expanded=False)

# ==========================================
# 6. 结果展示面板
# ==========================================

st.divider()

if not st.session_state.history:
    st.info("暂无数据，请点击开始调试。")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 结果详情", "📈 趋势分析", "📝 差异对比", "📟 运行日志"])
    
    with tab1:
        # 版本选择器
        versions = [f"v{h['version']} (Score: {h['evaluation']['score']}) {'↩️' if h['status']=='rolled_back' else ''}" for h in st.session_state.history]
        sel_idx = st.selectbox("选择版本查看", range(len(versions)), format_func=lambda x: versions[x])
        data = st.session_state.history[sel_idx]
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📝 提示词模板")
            st.code(data['prompt_template'], language='markdown')
            
            st.markdown("#### 🤖 实际输出")
            st.info(data['output'])
            
        with c2:
            st.markdown("#### 🏆 质量评估")
            score = data['evaluation']['score']
            st.metric("得分", score, delta=None)
            st.warning(f"**评价理由**: {data['evaluation']['reason']}")
            
            st.markdown("---")
            st.markdown("#### 🔧 人工干预")
            # 允许用户基于此版本修改
            manual_edit = st.text_area("在此版本基础上修改:", value=data['prompt_template'], key=f"manual_{sel_idx}")
            if st.button("采纳此修改并覆盖输入框", key=f"btn_{sel_idx}"):
                st.session_state.current_prompt = manual_edit
                st.rerun()

    with tab2:
        if len(st.session_state.history) > 0:
            scores = [h['evaluation']['score'] for h in st.session_state.history]
            st.line_chart(scores)
            
            # 数据表导出
            import pandas as pd
            df = pd.DataFrame([{
                "Version": h['version'],
                "Score": h['evaluation']['score'],
                "Output": h['output'][:50]+"...",
                "Status": h['status']
            } for h in st.session_state.history])
            st.dataframe(df, use_container_width=True)

    with tab3:
        if len(st.session_state.history) >= 2:
            v_a = st.selectbox("版本 A", range(len(versions)), index=len(versions)-2, key="diff_a")
            v_b = st.selectbox("版本 B", range(len(versions)), index=len(versions)-1, key="diff_b")
            
            txt_a = st.session_state.history[v_a]['prompt_template']
            txt_b = st.session_state.history[v_b]['prompt_template']
            
            st.markdown("### 差异视图 (A vs B)")
            st.markdown(get_diff_html(txt_a, txt_b), unsafe_allow_html=True)
        else:
            st.warning("需要至少两个版本才能进行对比")

    with tab4:
        st.text_area("系统日志", value="\n".join(st.session_state.logs), height=300)

    # 导出按钮
    report_data = json.dumps(st.session_state.history, indent=2, ensure_ascii=False)
    st.download_button("💾 导出完整报告 (JSON)", report_data, file_name="debug_report.json", mime="application/json")