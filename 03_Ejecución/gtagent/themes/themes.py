THEMES = {
  "factory": {
    "header_bg": "#003366",
    "header_color": "#ffffff",
    "user_bg": "#d0ebff",
    "agent_bg": "#d3f9d8",
    "container_bg": "#f7f7f7",
    "accent": "#1f77b4",
    "font": "sans-serif"
  },
  "law_office": {
    "header_bg": "#2c2c2c",
    "header_color": "#f0f0f0",
    "user_bg": "#f5f5dc",
    "agent_bg": "#e6e6fa",
    "container_bg": "#fafafa",
    "accent": "#8b0000",
    "font": "serif"
  },
  "blue_tech": {
    "header_bg": "#0f2027",
    "header_color": "#ffffff",
    "user_bg": "#4facfe",
    "agent_bg": "#00f2fe",
    "container_bg": "#e0f7fa",
    "accent": "#1976d2",
    "font": "Roboto, sans-serif"
  },
  "dark_mode": {
    "header_bg": "#121212",
    "header_color": "#e0e0e0",
    "user_bg": "#1e88e5",
    "agent_bg": "#43a047",
    "container_bg": "#2a2a2a",
    "accent": "#90caf9",
    "font": "Segoe UI, sans-serif"
  }
}

def apply_theme(theme):
  import streamlit as st
  st.markdown(f"""
    <style>
    .app-header {{
      background-color: {theme['header_bg']};
      color: {theme['header_color']};
      font-family: {theme['font']};
      font-size: 28px;
      padding: 0.5em 1em;
      border-radius: 10px;
      margin-bottom: 1em;
    }}
    .chat-container {{
      max-height: 400px;
      overflow-y: auto;
      padding: 1em;
      background-color: {theme['container_bg']};
      border-radius: 10px;
      border: 1px solid #ccc;
    }}
    .chat-user {{
      text-align: right;
      background-color: {theme['user_bg']};
      padding: 0.5em;
      margin: 0.3em;
      border-radius: 10px;
    }}
    .chat-agent {{
      text-align: left;
      background-color: {theme['agent_bg']};
      padding: 0.5em;
      margin: 0.3em;
      border-radius: 10px;
    }}
    </style>
  """, unsafe_allow_html=True)
