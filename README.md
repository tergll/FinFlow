# AI-Powered Financial Research Assistant

## Overview

The AI-Powered Financial Research Assistant transforms how financial professionals conduct research, analyze data, and present findings. Our solution combines advanced AI tools to automate the entire research workflow, allowing professionals to focus on decision-making rather than data gathering.

## Key Features

### 🔍 Automated Research
- Comprehensive web search for financial information, news, and market insights
- Real-time data collection from multiple sources
- Customizable research parameters for targeted analysis

### 📊 Data Analysis
- Intelligent processing of structured and unstructured data
- Identification of market trends and patterns
- Statistical analysis with visualized results

### 📁 Document Integration
- Seamless upload and analysis of PDFs, Excel sheets, and other documents
- Extraction of key insights from proprietary data
- Contextual integration with web-sourced information

### 🎙️ Professional Presentation Generation
- Automated creation of speech scripts with key findings
- AI voice synthesis for professional narration
- Presentation-ready content formatted for business contexts

## Technology Stack

### Core Technologies
- **Python**: Primary development language
- **Streamlit**: Interactive web application framework

### AI & Data Processing
- **Gemini**: Advanced AI model for reasoning and analysis
- **Apify**: Web scraping and data collection
- **ElevenLabs**: Voice synthesis and audio generation

### Infrastructure
- **Render**: Cloud deployment and hosting
- **Windsurf**: Development and optimization tools
- **Tavus**: AI presenter integration capabilities

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/financial-research-assistant.git

# Navigate to the project directory
cd financial-research-assistant

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys to the .env file

# Launch the application
streamlit run app.py
```

<img src="FinFlow_Demo.gif" alt="FinFlow Demo" width="100%">


## Usage Guide

1. **Start Research**: Enter a financial topic, company name, or research question in the chat interface
2. **Select Sources**: Choose which data sources to include in your research
3. **Upload Documents**: Add proprietary documents to enhance the analysis (optional)
4. **Generate Insights**: The system will automatically gather, analyze, and synthesize information
5. **Review Results**: Explore detailed research findings, complete with citations and sources
6. **Generate Presentation**: Create a professional speech script and audio presentation with a single click

## Project Architecture

Our solution integrates multiple systems to create a seamless research experience:

1. **Research Planning**: Analyzes the query to determine optimal research strategy
2. **Data Collection**: Gathers information from web sources and uploaded documents
3. **Analysis Engine**: Processes collected data to extract meaningful insights
4. **Synthesis Layer**: Combines findings into a coherent narrative
5. **Presentation Generator**: Creates professional output for sharing findings

## Future Enhancements

- Integration with financial data APIs for real-time market information
- Expanded document analysis capabilities for complex financial reports
- Enhanced visualization tools for more interactive data presentation
- Collaborative features for team-based financial research

## Team

Developed for the AI Agents Hackathon at NVIDIA GTC 2025, organized by Vertex Ventures US and CreatorsCorner.

## Acknowledgments

Special thanks to the hackathon sponsors whose technologies made this project possible.