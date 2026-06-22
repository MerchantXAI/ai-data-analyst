import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import io
import streamlit.components.v1 as components
from google.api_core import exceptions  # Added to catch the rate limit securely

# Secure connection to the Gemini API Key vault
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Missing Gemini API Key. Please add it to your Streamlit Secrets.")

st.set_page_config(page_title="AI Audio Analyst", layout="wide")

st.title("🎙️ Multi-Modal AI Data Agent & Animated Dashboard")
st.write("Upload a corporate dataset, view animated trend-lines, and talk directly to your metrics using the physical microphone button below.")

# Maintain session states for cloud execution
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_dataframe" not in st.session_state:
    st.session_state.active_dataframe = None

# Sidebar Controls
with st.sidebar:
    st.header("📂 Data Control Center")
    uploaded_file = st.file_uploader("Upload Corporate CSV", type=["csv"])
    
    if st.button("💡 Load Demo Sales Sheet"):
        sample_csv = """Month,Sales,AdSpend,NewCustomers
        January,45000,5000,120
        February,52000,5500,145
        March,48000,6000,110
        April,61000,7000,190
        May,74000,8500,240
        June,93000,9000,310"""
        st.session_state.active_dataframe = pd.read_csv(io.StringIO(sample_csv))
        st.success("Demo sales data loaded!")

if uploaded_file is not None:
    st.session_state.active_dataframe = pd.read_csv(uploaded_file)

# Process and Render Dashboard Elements
if st.session_state.active_dataframe is not None:
    df = st.session_state.active_dataframe
    
    st.subheader("📈 Interactive Automated Visualizations")
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(num_cols) >= 1:
        x_axis = text_cols[0] if text_cols else df.columns[0]
        y_axis = num_cols[0]
        
        fig = px.line(df, x=x_axis, y=y_axis, markers=True,
                      title=f"Performance Matrix: {y_axis} over time",
                      template="plotly_dark")
        fig.update_layout(transition_duration=400)
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("---")
    
    # Dual-Input Layout: Split screen into Voice and Text options
    st.subheader("🔊 Command Center: Speak or Chat with Your Data")
    col1, col2 = st.columns([1, 1])
    
    user_query = None
    audio_data_payload = None

    with col1:
        st.markdown("**Option A: Use Physical Microphone**")
        voice_file = st.audio_input("Click the record icon below to speak to your dashboard:")
        if voice_file is not None:
            audio_data_payload = {
                "mime_type": voice_file.type,
                "data": voice_file.read()
            }
            user_query = "[Sent Spoken Voice Command]"

    with col2:
        st.markdown("**Option B: Type Text Instead**")
        text_input = st.text_input("Type your analytical question here:", placeholder="e.g., Summarize our performance vectors.")
        if st.button("Submit Text Command") and text_input:
            user_query = text_input

    # Execution Loop for AI processing
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        data_string_context = df.to_string()
        agent_instruction = f"""You are an elite operational corporate data scientist speaking directly to the business owner. 
        Here is the data currently active on their screen:\n\n{data_string_context}\n\n
        Answer the user's input request clearly using explicit numbers from the data. Keep your response to 2-3 impact-focused sentences."""
        
        with st.chat_message("assistant"):
            with st.spinner("Processing dashboard telemetry..."):
                try:
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    
                    if audio_data_payload:
                        response = model.generate_content([audio_data_payload, agent_instruction])
                    else:
                        response = model.generate_content(agent_instruction + f"\n\nUser Question: {user_query}")
                        
                    st.markdown(response.text)
                    
                    # Clean markdown formatting so the browser reads it smoothly
                    spoken_text = response.text.replace("*", "").replace("#", "").replace("-", "")
                    
                    tts_script = f"""
                    <script>
                        if ('speechSynthesis' in window) {{
                            window.speechSynthesis.cancel(); // Stop any previous speech
                            var utterance = new SpeechSynthesisUtterance({repr(spoken_text)});
                            utterance.rate = 1.0; 
                            window.speechSynthesis.speak(utterance);
                        }}
                    </script>
                    """
                    components.html(tts_script, height=0)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                
                # --- CRASH PROTECTION BLOCK ---
                except exceptions.ResourceExhausted:
                    st.error("⏳ **Google API Rate Limit Reached.** The free tier has a temporary limit on how much data can be processed per minute. Please wait 60 seconds and click or speak your command again!")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                # -------------------------------
        
    if st.session_state.chat_history:
        with st.expander("📚 View Complete Session Chat History Log"):
            for msg in st.session_state.chat_history:
                st.write(f"**{msg['role'].capitalize()}**: {msg['content']}")
else:
    st.info("Waiting for telemetry. Drop a CSV file or tap 'Load Demo Sales Sheet' in the sidebar to activate the system.")
