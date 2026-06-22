import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import io

# Securely pull the API Key from Streamlit Secrets
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Missing Gemini API Key. Please add it to your Streamlit Secrets.")

st.set_page_config(page_title="AI Voice Analyst", layout="wide")

st.title("🎙️ Voice-Activated AI Data Agent & Animated Dashboard")
st.write("Upload a CSV file or load sample data. Use your keyboard or browser microphone to chat directly with your visual data.")

# Initialize chat history and data state in cloud memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None

# Sidebar for data input options
with st.sidebar:
    st.header("📂 Data Control Center")
    uploaded_file = st.file_uploader("Upload Corporate CSV", type=["csv"])
    
    # Golden Feature: Demo button so prospects can test the app instantly without their own file
    if st.button("💡 Load Demo Sales Data"):
        demo_data = """Month,Sales,AdSpend,NewCustomers
        January,45000,5000,120
        February,52000,5500,145
        March,48000,6000,110
        April,61000,7000,190
        May,74000,8500,240
        June,93000,9000,310"""
        st.session_state.df = pd.read_csv(io.StringIO(demo_data))
        st.success("Demo data loaded!")

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)

# If data exists, render the dashboard and animated visuals
if st.session_state.df is not None:
    df = st.session_state.df
    
    st.subheader("📈 Real-Time Automated Visualizations")
    
    # Automatically identify numerical columns for plotting
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(num_cols) >= 1:
        x_axis = text_cols[0] if text_cols else df.columns[0]
        y_axis = num_cols[0]
        
        # Generate an interactive, animated vector chart using Plotly
        fig = px.bar(df, x=x_axis, y=y_axis, color=y_axis,
                     title=f"Trend Analysis: {y_axis} over {x_axis}",
                     labels={y_axis: y_axis, x_axis: x_axis},
                     template="plotly_dark")
        
        fig.update_layout(transition_duration=500) # Smooth animation transition
        st.plotly_chart(fig, use_container_width=True)
        
        # Display raw data snapshot for context
        with st.expander("👀 View Full Data Table"):
            st.dataframe(df, use_container_width=True)
            
    st.markdown("---")
    st.subheader("💬 Talk to Your Dashboard")
    st.caption("Pro-Tip: Click the microphone icon on your device's keyboard or browser input to speak your command directly!")

    # Display running chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User chat/voice input command
    if user_query := st.chat_input("Ask something (e.g., 'Which month had the highest sales ROI?')"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Package the active data context into the LLM payload
        data_context = df.to_string()
        full_agent_prompt = f"""You are an expert corporate data scientist speaking directly to the CEO. 
        Here is the data currently displayed on the dashboard above:\n\n{data_context}\n\n
        The user is asking this question about the visible dashboard: '{user_query}'\n\n
        Answer clearly in 2-3 sentences. reference explicit metrics seen in the data."""
        
        # Execute processing via Gemini 2.5 Flash
        with st.chat_message("assistant"):
            with st.spinner("Analyzing dashboard context..."):
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(full_agent_prompt)
                st.markdown(response.text)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response.text})
else:
    st.info("Please upload a CSV file or click 'Load Demo Sales Data' in the sidebar to activate the dashboard.")
