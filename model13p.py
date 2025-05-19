import streamlit as st
import boto3
import json
import tempfile
import os
from botocore.exceptions import NoCredentialsError, ClientError
from PyPDF2 import PdfReader
import uuid
from datetime import datetime
from io import BytesIO

# ==== STREAMLIT APP CONFIGURATION ====
st.set_page_config(
    page_title="AI-Powered Text Analysis Assistant",
    page_icon="📝",
    layout="wide"
)

# ==== APPLICATION SETTINGS ====
# Make bucket name configurable
DEFAULT_S3_BUCKET = 'pruebafinal1'  

# ==== AWS SETUP AND CONFIGURATION ====
def setup_aws_clients():
    """Set up and test AWS clients with proper error handling"""
    st.sidebar.header("AWS Configuration")
    
    # Get AWS region
    aws_region = st.sidebar.text_input("AWS Region", "us-east-1")
    
    # S3 bucket configuration
    s3_bucket = st.sidebar.text_input("S3 Bucket Name", DEFAULT_S3_BUCKET)
    
    # Initialize session and clients
    try:
        if st.sidebar.checkbox("Use AWS Profile", value=True):
            profile_name = st.sidebar.text_input("AWS Profile Name", "recruitment-assistant")
            session = boto3.Session(profile_name=profile_name, region_name=aws_region)
        else:
            # Use environment variables or default credentials
            session = boto3.Session(region_name=aws_region)
        
        # Initialize S3 client
        s3 = session.client("s3")
        
        # Initialize Bedrock clients - separate clients for runtime and management
        bedrock_runtime = None
        bedrock = None
        
        # Check if bedrock is available in this region
        available_services = session.get_available_services()
        
        if "bedrock-runtime" in available_services:
            bedrock_runtime = session.client("bedrock-runtime")
        else:
            st.sidebar.warning(f"⚠️ Bedrock Runtime not available in region {aws_region}")
            
        if "bedrock" in available_services:
            bedrock = session.client("bedrock")
        else:
            st.sidebar.warning(f"⚠️ Bedrock management API not available in region {aws_region}")
            
        # Test S3 connection
        if st.sidebar.button("Test S3 Connection"):
            try:
                # First check if bucket exists
                response = s3.list_buckets()
                buckets = [bucket['Name'] for bucket in response['Buckets']]
                
                if s3_bucket in buckets:
                    # Try to access the bucket
                    s3.head_bucket(Bucket=s3_bucket)
                    st.sidebar.success(f"✅ S3 connection successful to bucket {s3_bucket}")
                else:
                    st.sidebar.warning(f"⚠️ Bucket {s3_bucket} does not exist. Do you want to create it?")
                    if st.sidebar.button("Create Bucket"):
                        try:
                            # Create bucket with appropriate configuration
                            if aws_region == "us-east-1":
                                s3.create_bucket(Bucket=s3_bucket)
                            else:
                                s3.create_bucket(
                                    Bucket=s3_bucket,
                                    CreateBucketConfiguration={'LocationConstraint': aws_region}
                                )
                            st.sidebar.success(f"✅ Bucket {s3_bucket} created successfully!")
                        except Exception as e:
                            st.sidebar.error(f"❌ Failed to create bucket: {str(e)}")
                    
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                if error_code == '404':
                    st.sidebar.error(f"❌ Bucket {s3_bucket} does not exist")
                elif error_code == '403':
                    st.sidebar.error(f"❌ No permission to access bucket {s3_bucket}")
                else:
                    st.sidebar.error(f"❌ S3 connection error: {error_code} - {str(e)}")
            except Exception as e:
                st.sidebar.error(f"❌ S3 connection failed: {str(e)}")
                
        # Display available bedrock models (if connected)
        if bedrock and st.sidebar.button("List Available Bedrock Models"):
            try:
                response = bedrock.list_foundation_models()
                model_list = [model.get('modelId') for model in response.get('modelSummaries', [])[:10]]
                
                if model_list:
                    st.sidebar.success("✅ Found the following Bedrock models:")
                    for model in model_list:
                        st.sidebar.info(f"- {model}")
                else:
                    st.sidebar.warning("No models found or you don't have access to any models")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error listing models: {str(e)}")
                
        # Let user select a model if bedrock is available
        model_id = None
        if bedrock_runtime and bedrock:
            try:
                # Fetch list of models from Bedrock
                response = bedrock.list_foundation_models()
                model_summaries = response.get('modelSummaries', [])
                model_names = [model.get('modelId') for model in model_summaries]

                # Check if models are available
                if model_names:
                    selected_model = st.sidebar.selectbox(
                        "Select Bedrock Model",
                        ["Choose a model"] + model_names
                    )
                    if selected_model != "Choose a model":
                        model_id = selected_model
                    else:
                        model_id = None
                else:
                    st.sidebar.write("No available models found. Please check your AWS account and region.")
                    model_id = None
            except:
                # Fallback to common models if we can't fetch the list
                common_models = [
                    "Choose a model",
                    "amazon.titan-text-express-v1",
                    "anthropic.claude-3-haiku-20240307-v1",
                    "anthropic.claude-3-sonnet-20240229-v1",
                    "anthropic.claude-instant-v1",
                    "ai21.j2-ultra-v1",
                    "meta.llama2-13b-chat-v1"
                ]
                selected_model = st.sidebar.selectbox("Select Bedrock Model", common_models)
                if selected_model != "Choose a model":
                    model_id = selected_model
                else:
                    model_id = None
                
        return session, s3, bedrock_runtime, bedrock, model_id, aws_region, s3_bucket
    
    except NoCredentialsError:
        st.sidebar.error("❌ No AWS credentials found. Please configure AWS credentials.")
        return None, None, None, None, None, aws_region, s3_bucket
    except Exception as e:
        st.sidebar.error(f"❌ Error setting up AWS: {str(e)}")
        return None, None, None, None, None, aws_region, s3_bucket

# Initialize AWS clients
session, s3, bedrock_runtime, bedrock, MODEL_ID, AWS_REGION, S3_BUCKET = setup_aws_clients()

# ==== MAIN CONTENT ====
st.title("AI-Powered Text Analysis Assistant")

# Use tabs for different functionality
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "User Own Text", 
    "Change Text Tone", 
    "Topic Idea Explorer", 
    "Text Evaluation", 
    "LaTeX Maker", 
    "Reference Maker", 
    "Setup Help"
])

# ==== HELPER FUNCTION FOR BEDROCK REQUESTS ====
def call_bedrock_model(prompt, model_id):
    """Generic function to call Bedrock models with proper error handling"""
    if not bedrock_runtime or not model_id:
        return "⚠️ Bedrock runtime client not available or no model selected. Please configure AWS properly."
    
    try:
        # Check which model family we're using
        if "claude" in model_id.lower():
            # Claude models
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        elif "titan" in model_id.lower():
            # Amazon Titan models
            body = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 2000,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            })
        elif "llama" in model_id.lower():
            # Meta Llama models
            body = json.dumps({
                "prompt": prompt,
                "max_gen_len": 2000,
                "temperature": 0.7,
                "top_p": 0.9
            })
        else:
            # Generic format for other models
            body = json.dumps({
                "prompt": prompt,
                "max_tokens": 2000
            })

        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body,
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get('body').read().decode('utf-8'))
        
        # Extract response based on model
        if "claude" in model_id.lower():
            # Claude models response format
            return response_body.get("content", [{}])[0].get("text", "No content returned")
        elif "titan" in model_id.lower():
            # Titan models response format
            return response_body.get("results", [{}])[0].get("outputText", "No content returned")
        else:
            # Generic fallback - try common response formats
            if "generated_text" in response_body:
                return response_body["generated_text"]
            elif "completion" in response_body:
                return response_body["completion"]
            else:
                # Return the whole response for debugging
                return f"Response received but format unknown: {json.dumps(response_body, indent=2)}"
            
    except Exception as e:
        return f"Error calling model: {str(e)}\n\nTroubleshooting tips:\n1. Check if you have access to the selected model\n2. Verify your AWS credentials have proper permissions\n3. Make sure Bedrock is available in your region"

# ==== TAB 1: USER OWN TEXT (ENHANCED) ====
with tab1:
    st.header("Analyze Your Own Text")
    st.write("Input your text and get a comprehensive analysis with different components and corrections.")
    
    # Input text area
    user_text = st.text_area("Enter your text for analysis:", height=200, key="user_text_input")
    
    # Function to analyze text and generate components
    def analyze_user_text_enhanced(text, model_id):
        sections = [
            "hypothesis", "main_bullet_points", "most_important_data_points", 
            "summary", "abstract", "introduction", "body_text", "conclusion", "appendix"
        ]
        
        results = {}
        
        for section in sections:
            prompt = f"""
            Analyze the following text and generate the {section.replace('_', ' ')} section.
            
            Text to analyze:
            {text}
            
            If the text doesn't contain a clear {section.replace('_', ' ')}, respond with: "This text doesn't contain a {section.replace('_', ' ')}."
            
            Otherwise, please provide only the {section.replace('_', ' ')} for this text. Be concise and relevant.
            """
            
            result = call_bedrock_model(prompt, model_id)
            results[section] = result
        
        # Add coherence and style analysis
        corrections_prompt = f"""
        Analyze the following text for various corrections and improvements:
        
        Text to analyze:
        {text}
        
        Please provide analysis in the following format:
        
        SPELLING CORRECTIONS:
        [Evaluate spelling, identify errors, and provide suggestions]
        
        GRAMMAR CORRECTIONS:
        [Evaluate grammar, identify errors, and provide suggestions]
        
        COHERENCE CORRECTIONS:
        [Evaluate coherence, identify errors, and provide suggestions]
        
        STYLE CORRECTIONS:
        [Evaluate style, identify errors, and provide suggestions]
        
        ORDER CORRECTIONS:
        [Evaluate the order of ideas in the document, identify errors, and provide suggestions]
        
        PROPOSED CORRECTION:
        [Provide a corrected version of the text that addresses all the above issues to make it clearer]
        """
        
        corrections = call_bedrock_model(corrections_prompt, model_id)
        
        # Parse corrections into separate sections using a more robust approach
        correction_sections = {}
        sections_to_extract = [
            ("SPELLING CORRECTIONS:", "spelling"),
            ("GRAMMAR CORRECTIONS:", "grammar"),
            ("COHERENCE CORRECTIONS:", "coherence"),
            ("STYLE CORRECTIONS:", "style"),
            ("ORDER CORRECTIONS:", "order"),
            ("PROPOSED CORRECTION:", "proposed")
        ]
        
        # Create a list of all found markers with their positions
        found_markers = []
        for marker, key in sections_to_extract:
            pos = corrections.find(marker)
            if pos != -1:
                found_markers.append((pos, marker, key))
        
        # Sort by position to process them in order
        found_markers.sort(key=lambda x: x[0])
        
        # Extract content between markers
        for i, (pos, marker, key) in enumerate(found_markers):
            start_idx = pos + len(marker)
            
            # Find the end position (next marker or end of text)
            if i + 1 < len(found_markers):
                end_idx = found_markers[i + 1][0]
                correction_sections[key] = corrections[start_idx:end_idx].strip()
            else:
                # This is the last marker, take everything to the end
                correction_sections[key] = corrections[start_idx:].strip()
        
        # Fill in any missing sections
        for _, key in sections_to_extract:
            if key not in correction_sections:
                correction_sections[key] = "No corrections needed."
        
        results["corrections"] = correction_sections
        return results
    
    # Generate analysis button
    if st.button("Analyze Text", key="analyze_text"):
        if not user_text.strip():
            st.error("Please enter some text to analyze.")
        elif not MODEL_ID:
            st.error("⚠️ Please select a model in the sidebar first")
        else:
            with st.spinner("Analyzing your text..."):
                analysis_results = analyze_user_text_enhanced(user_text, MODEL_ID)
                st.session_state["analysis_results"] = analysis_results
    
    # Display results if available
    if "analysis_results" in st.session_state:
        st.subheader("Analysis Results")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        sections = [
            ("hypothesis", "Hypothesis"),
            ("main_bullet_points", "Main Bullet Points"),
            ("most_important_data_points", "Most Important Data Points"),
            ("summary", "Summary"),
            ("abstract", "Abstract"),
            ("introduction", "Introduction"),
            ("body_text", "Body Text"),
            ("conclusion", "Conclusion"),
            ("appendix", "Appendix")
        ]
        
        for i, (key, title) in enumerate(sections):
            with col1 if i % 2 == 0 else col2:
                st.text_area(
                    title,
                    value=st.session_state["analysis_results"].get(key, ""),
                    height=150,
                    key=f"result_{key}"
                )
        
        # Display corrections in separate boxes
        if "corrections" in st.session_state["analysis_results"]:
            st.subheader("Text Corrections and Improvements")
            
            correction_types = [
                ("spelling", "Spelling Corrections"),
                ("grammar", "Grammar Corrections"),
                ("coherence", "Coherence Corrections"),
                ("style", "Style Corrections"),
                ("order", "Order Corrections"),
                ("proposed", "Proposed Corrected Text")
            ]
            
            # Create two columns for corrections
            col1, col2 = st.columns(2)
            
            for i, (key, title) in enumerate(correction_types):
                with col1 if i % 2 == 0 else col2:
                    st.text_area(
                        title,
                        value=st.session_state["analysis_results"]["corrections"].get(key, ""),
                        height=150,
                        key=f"correction_{key}"
                    )


# ==== TAB 2: CHANGE TEXT TONE (ENHANCED) ====
with tab2:
    st.header("Change Text Tone")
    st.write("Transform your text into different tones and styles with advanced customization options.")
    
    # Create columns for controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Tone selection
        tone_options = ["Academic", "Technical", "Simple", "Descriptive", "Narrative"]
        selected_tone = st.selectbox("Select the tone you want:", tone_options)
        
        # Text type selection
        text_type_options = ["Report", "Summary", "Academic Paper", "Press Release"]
        selected_text_type = st.selectbox("Select the type of text:", text_type_options)
    
    with col2:
        # Technical level
        technical_level = st.selectbox(
            "Technical vocabulary level:",
            ["Very Low", "Low", "Moderate", "High", "Very High"]
        )
        
        # Formality level
        formality_level = st.selectbox(
            "Formality level:",
            ["Very Low", "Low", "Moderate", "High", "Very High"]
        )
        
        # Use of numbers and statistics
        statistics_level = st.selectbox(
            "Use of numbers and statistics:",
            ["Very Low", "Low", "Moderate", "High", "Very High"]
        )
    
    # Input text area
    tone_text = st.text_area("Enter your text to change tone:", height=200, key="tone_text_input")
    
    # Function to change text tone for specific section with enhanced options
    def change_text_tone_section_enhanced(text, tone, section, text_type, technical_level, formality_level, statistics_level, model_id):
        prompt = f"""
        Transform the following text according to these specifications, if the text is a code, make a text with the following structure about the code:
        
        Style: {tone}
        Text Type: {text_type}
        Technical Vocabulary Level: {technical_level}
        Formality Level: {formality_level}
        Use of Numbers and Statistics: {statistics_level}
        
        Generate the {section.replace('_', ' ')} section for this text type.
        
        Original text:
        {text}
        
        Instructions:
        - Write in a {tone.lower()} style appropriate for a {text_type.lower()}
        - Use {technical_level.lower()} level technical vocabulary
        - Maintain {formality_level.lower()} formality
        - Include {statistics_level.lower()} level of numerical data and statistics
        - Structure appropriately for a {text_type.lower()}
        - If the section doesn't apply to this text type, respond with: "This section doesn't apply to a {text_type.lower()}."
        
        Provide only the {section.replace('_', ' ')} portion.
        """
        
        return call_bedrock_model(prompt, model_id)
    
    # Function to transform the text
    def transform_text(text, tone, text_type, technical_level, formality_level, statistics_level, model_id):
        prompt = f"""
        Transform the following text according to these specifications:
        
        Style: {tone}
        Text Type: {text_type}
        Technical Vocabulary Level: {technical_level}
        Formality Level: {formality_level}
        Use of Numbers and Statistics: {statistics_level}
        
        Original text:
        {text}
        
        Instructions:
        - Write in a {tone.lower()} style appropriate for a {text_type.lower()}
        - Use {technical_level.lower()} level technical vocabulary
        - Maintain {formality_level.lower()} formality
        - Include {statistics_level.lower()} level of numerical data and statistics
        - Structure appropriately for a {text_type.lower()}
        
        Please provide the transformed text.
        """
        
        return call_bedrock_model(prompt, model_id)
    
    # Transform text button
    if st.button("Transform Text", key="transform_text_button"):
        if not tone_text.strip():
            st.error("Please enter some text to transform.")
        elif not MODEL_ID:
            st.error("⚠️ Please select a model in the sidebar first")
        else:
            with st.spinner("Transforming your text..."):
                transformed_text = transform_text(
                    tone_text, 
                    selected_tone, 
                    selected_text_type, 
                    technical_level, 
                    formality_level, 
                    statistics_level, 
                    MODEL_ID
                )
                st.session_state["transformed_text"] = transformed_text
    
    # Display transformed text if available
    if "transformed_text" in st.session_state:
        st.subheader("Transformed Text")
        st.text_area(
            "Result",
            value=st.session_state["transformed_text"],
            height=300,
            key="transform_result"
        )

# ==== TAB 4: TEXT EVALUATION ====
with tab4:
    st.header("Text Evaluation")
    st.write("Evaluate your text for style, grammar, and other aspects.")
    
    # Input text area - using a unique key to avoid conflicts
    evaluation_text = st.text_area("Enter your text to evaluate:", height=200, key="text_evaluation_input")
    
    # Function to evaluate text
    def evaluate_text_comprehensive(text, model_id):
        prompt = f"""
        Provide a comprehensive evaluation of the following text:
        
        Text to evaluate:
        {text}
        
        Please structure your evaluation in the following format (and give the corresponding scores out of 10, followed by proposed corrections to address the issues):
        
        SPELLING EVALUATION:
        [Evaluate spelling, identify errors, and provide suggestions]
        
        GRAMMAR EVALUATION:
        [Evaluate grammar, identify errors, and provide suggestions]
        
        STYLE EVALUATION:
        [Evaluate writing style, flow, and clarity]
        
        COHERENCE EVALUATION:
        [Evaluate logical flow and connection between ideas]
        
        OVERALL EVALUATION:
        [Provide an overall assessment of the text quality]
        """
        
        return call_bedrock_model(prompt, model_id)
    
    # Function to parse evaluation results
    def parse_evaluation_results(evaluation_text):
        sections = {}
        
        # Define the sections to extract
        section_markers = [
            ("SPELLING EVALUATION:", "spelling"),
            ("GRAMMAR EVALUATION:", "grammar"),
            ("STYLE EVALUATION:", "style"),
            ("COHERENCE EVALUATION:", "coherence"),
            ("OVERALL EVALUATION:", "overall")
        ]
        
        for marker, key in section_markers:
            if marker in evaluation_text:
                start_idx = evaluation_text.find(marker) + len(marker)
                # Find the next section or end of text
                next_markers = [m for m, _ in section_markers if m != marker and m in evaluation_text]
                if next_markers:
                    next_indices = [evaluation_text.find(m) for m in next_markers if evaluation_text.find(m) > start_idx]
                    if next_indices:
                        next_idx = min(next_indices)
                        sections[key] = evaluation_text[start_idx:next_idx].strip()
                    else:
                        sections[key] = evaluation_text[start_idx:].strip()
                else:
                    sections[key] = evaluation_text[start_idx:].strip()
            else:
                sections[key] = "No evaluation available for this section."
        
        return sections
    
    # Evaluate text button - using a unique key to avoid conflict
    if st.button("Evaluate Text", key="evaluate_text_button"):
        if not evaluation_text.strip():
            st.error("Please enter some text to evaluate.")
        elif not MODEL_ID:
            st.error("⚠️ Please select a model in the sidebar first")
        else:
            with st.spinner("Evaluating your text..."):
                evaluation_results = evaluate_text_comprehensive(evaluation_text, MODEL_ID)
                parsed_results = parse_evaluation_results(evaluation_results)
                st.session_state["evaluation_results"] = parsed_results
                st.session_state["raw_evaluation"] = evaluation_results
    
    # Display evaluation results
    if "evaluation_results" in st.session_state:
        st.subheader("Text Evaluation Results")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        evaluation_sections = [
            ("spelling", "Spelling Evaluation"),
            ("grammar", "Grammar Evaluation"),
            ("style", "Style Evaluation"),
            ("coherence", "Coherence Evaluation"),
            ("overall", "Overall Evaluation")
        ]
        
        for i, (key, title) in enumerate(evaluation_sections):
            with col1 if i % 2 == 0 else col2:
                if key in st.session_state["evaluation_results"]:
                    st.text_area(
                        title,
                        value=st.session_state["evaluation_results"][key],
                        height=200,
                        key=f"eval_{key}"
                    )
                else:
                    st.text_area(
                        title,
                        value="No evaluation available for this section.",
                        height=200,
                        key=f"eval_{key}_empty"
                    )

# ==== TAB 5: LATEX MAKER ====
with tab5:
    st.header("LaTeX Maker")
    st.write("Convert your text into properly formatted LaTeX code for RMarkdown that can be knitted into a PDF.")
    
    # Text type selection for LaTeX
    latex_text_type = st.selectbox(
        "Select the type of document:",
        ["Academic Paper", "Report", "Press Release", "Technical Manual", "Thesis", "Article"],
        key="latex_text_type"
    )
    
    # Input text area
    latex_text = st.text_area("Enter your text to convert to LaTeX:", height=200, key="latex_text_input")
    
    # Function to generate LaTeX code
    def generate_latex_code(text, document_type, model_id):
        prompt = f"""
        Convert the following text into properly formatted LaTeX code suitable for RMarkdown that can be knitted into a PDF.
        
        Document Type: {document_type}
        
        Text to convert:
        {text}
        
        Requirements:
        1. Create a complete LaTeX document structure appropriate for a {document_type.lower()}
        2. Include proper document class and packages
        3. Format any equations, numbers, or special formatting appropriately
        4. Add proper sectioning (\\section, \\subsection, etc.)
        5. Include title, author, date fields that can be customized
        6. Use proper LaTeX formatting for lists, emphasis, etc.
        7. Add comments explaining key formatting choices
        8. Make it compatible with RMarkdown output: pdf_document
        
        Provide only the LaTeX code that can be copied and pasted into RMarkdown.
        """
        
        return call_bedrock_model(prompt, model_id)
    
    # Generate LaTeX button
    if st.button("Generate LaTeX Code", key="generate_latex"):
        if not latex_text.strip():
            st.error("Please enter some text to convert.")
        elif not MODEL_ID:
            st.error("⚠️ Please select a model in the sidebar first")
        else:
            with st.spinner("Generating LaTeX code..."):
                latex_code = generate_latex_code(latex_text, latex_text_type, MODEL_ID)
                st.session_state["latex_code"] = latex_code
    
    # Display LaTeX code
    if "latex_code" in st.session_state:
        st.subheader("Generated LaTeX Code")
        st.text_area(
            "Copy this code into your RMarkdown document:",
            value=st.session_state["latex_code"],
            height=500,
            key="latex_output"
        )
        
        # Instructions for using the code
        st.info("""
        **How to use this LaTeX code:**
        1. Copy the generated code above
        2. Create a new RMarkdown file (.Rmd)
        3. Set the output to `pdf_document` in the YAML header
        4. Paste the LaTeX code in the document body
        5. Knit to PDF
        
        **Sample YAML header:**
        ```yaml
        ---
        title: "Your Title"
        author: "Your Name"
        date: "`r Sys.Date()`"
        output: pdf_document
        ---
        ```
        """)

# ==== TAB 6: REFERENCE MAKER ====
with tab6:
    st.header("Reference Maker")
    st.write("Create properly formatted references in APA, MLA, or Chicago style.")
    
    # Reference style selection
    reference_style = st.selectbox(
        "Select reference style:",
        ["APA", "MLA", "Chicago"],
        key="reference_style"
    )
    
    # Reference type selection
    reference_type = st.selectbox(
        "Select reference type:",
        ["Journal Article", "Book", "Website", "Conference Paper", "Thesis/Dissertation", "Report"],
        key="reference_type"
    )
    
    # Create input fields based on reference type
    col1, col2 = st.columns(2)
    
    with col1:
        author = st.text_input("Author(s):", key="ref_author")
        title = st.text_input("Title:", key="ref_title")
        year = st.text_input("Year Published:", key="ref_year")
        
        if reference_type in ["Journal Article", "Conference Paper"]:
            journal_conference = st.text_input("Journal/Conference Name:", key="ref_journal")
            volume = st.text_input("Volume:", key="ref_volume")
            issue = st.text_input("Issue:", key="ref_issue")
            pages = st.text_input("Pages:", key="ref_pages")
        
        if reference_type == "Book":
            publisher = st.text_input("Publisher:", key="ref_publisher")
            place_published = st.text_input("Place Published:", key="ref_place")
            edition = st.text_input("Edition (if not first):", key="ref_edition")
        
        if reference_type == "Website":
            website_name = st.text_input("Website Name:", key="ref_website")
            url = st.text_input("URL:", key="ref_url")
            access_date = st.text_input("Date Accessed:", key="ref_access_date")
    
    with col2:
        if reference_type == "Thesis/Dissertation":
            institution = st.text_input("Institution:", key="ref_institution")
            degree_type = st.text_input("Degree Type (Master's/PhD):", key="ref_degree")
            department = st.text_input("Department:", key="ref_department")
        
        if reference_type == "Report":
            organization = st.text_input("Organization:", key="ref_organization")
            report_number = st.text_input("Report Number:", key="ref_report_num")
            place_published = st.text_input("Place Published:", key="ref_report_place")
        
        # Additional fields for all types
        doi = st.text_input("DOI (if available):", key="ref_doi")
        notes = st.text_area("Additional Notes:", key="ref_notes", height=100)
    
    # Function to generate reference
    def generate_reference(style, ref_type, fields, model_id):
        # Create a string with all the provided information
        field_info = "\n".join([f"{key}: {value}" for key, value in fields.items() if value.strip()])
        
        prompt = f"""
        Create a properly formatted reference in {style} style for a {ref_type.lower()}.
        
        Reference Information:
        {field_info}
        
        Requirements:
        1. Follow {style} formatting guidelines exactly
        2. Include all provided information in the correct order
        3. Use proper punctuation, italics, and formatting
        4. Include DOI if provided
        5. Handle missing information appropriately
        6. Provide only the formatted reference
        
        Format the reference exactly as it should appear in a reference list.
        """
        
        return call_bedrock_model(prompt, model_id)
    
    # Generate reference button
    if st.button("Generate Reference", key="generate_reference"):
        if not author.strip() or not title.strip():
            st.error("Please provide at least author and title information.")
        elif not MODEL_ID:
            st.error("⚠️ Please select a model in the sidebar first")
        else:
            # Collect all field information
            fields = {
                "Author(s)": author,
                "Title": title,
                "Year": year,
                "DOI": doi,
                "Notes": notes
            }
            
            # Add type-specific fields
            if reference_type in ["Journal Article", "Conference Paper"]:
                fields["Journal/Conference"] = locals().get("journal_conference", "")
                fields["Volume"] = locals().get("volume", "")
                fields["Issue"] = locals().get("issue", "")
                fields["Pages"] = locals().get("pages", "")
            elif reference_type == "Book":
                fields["Publisher"] = locals().get("publisher", "")
                fields["Place Published"] = locals().get("place_published", "")
                fields["Edition"] = locals().get("edition", "")
            elif reference_type == "Website":
                fields["Website Name"] = locals().get("website_name", "")
                fields["URL"] = locals().get("url", "")
                fields["Date Accessed"] = locals().get("access_date", "")
            elif reference_type == "Thesis/Dissertation":
                fields["Institution"] = locals().get("institution", "")
                fields["Degree Type"] = locals().get("degree_type", "")
                fields["Department"] = locals().get("department", "")
            elif reference_type == "Report":
                fields["Organization"] = locals().get("organization", "")
                fields["Report Number"] = locals().get("report_number", "")
                fields["Place Published"] = locals().get("place_published", "")
            
            with st.spinner("Generating reference..."):
                formatted_reference = generate_reference(reference_style, reference_type, fields, MODEL_ID)
                st.session_state["formatted_reference"] = formatted_reference
    
    # Display generated reference
    if "formatted_reference" in st.session_state:
        st.subheader("Generated Reference")
        st.text_area(
            f"Copy this {reference_style} style reference:",
            value=st.session_state["formatted_reference"],
            height=150,
            key="reference_output"
        )
        
        st.success("✅ Reference generated successfully! You can copy and paste it into your document.")

# ==== TAB 7: SETUP HELP ====
with tab7:
    st.header("AWS Bedrock Setup Guide")
    
    st.markdown("""
    ## Troubleshooting AWS Bedrock Access
    
    If you're getting "AccessDeniedException" errors, follow these steps:
    
    ### 1. Verify AWS Region
    
    Bedrock is only available in specific regions:
    - US East (N. Virginia): `us-east-1`
    - US West (Oregon): `us-west-2`
    - Europe (Frankfurt): `eu-central-1`
    - Asia Pacific (Tokyo): `ap-northeast-1`
    
    Make sure you select one of these regions in the sidebar.
    
    ### 2. Enable Bedrock Model Access
    
    You need to request access to the specific models you want to use:
    
    1. Go to the [AWS Bedrock console](https://console.aws.amazon.com/bedrock/)
    2. Click on "Model access" in the left sidebar
    3. Click "Manage model access"
    4. Select the models you want to use (Claude, Titan, etc.)
    5. Click "Request model access"
    6. Wait for approval (some models are approved instantly)
    
    ### 3. Check IAM Permissions
    
    Ensure your IAM user/role has these permissions:
    
    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:ListFoundationModels",
                    "bedrock:GetFoundationModel"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock-runtime:InvokeModel"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucket",
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:CreateBucket",
                    "s3:ListAllMyBuckets"
                ],
                "Resource": [
                    "arn:aws:s3:::*",
                    "arn:aws:s3:::*/*"
                ]
            }
        ]
    }
    ```
    
    ### 4. Configure AWS CLI
    
    Make sure your AWS CLI is properly configured:
    
    ```bash
    aws configure --profile recruitment-assistant
    ```
    
    Enter your AWS Access Key ID, Secret Access Key, region (e.g., us-east-1), and output format (json).
    
    ### 5. Test Your Access
    
    Use these commands to test if you can list Bedrock models:
    
    ```bash
    aws bedrock list-foundation-models --region us-east-1 --profile recruitment-assistant
    ```
    
    Test S3:
    
    ```bash
    aws s3 ls --profile recruitment-assistant
    ```
    
    If these work, you should be able to access these services through this app.
    
    ### 6. Application Features
    
    This enhanced version includes:
    
    **Tab 1 - User Own Text:**
    - Comprehensive text analysis with sections detection
    - Detailed corrections for spelling, grammar, coherence, style, and order
    - Proposed corrected version of your text
    
    **Tab 2 - Change Text Tone:**
    - Advanced tone transformation with customizable parameters
    - Technical level, formality, and statistics usage controls
    - Text type selection (report, academic paper, etc.)
    - Coherence and style analysis of transformed text
    
    **Tab 3 - Topic Idea Explorer:**
    - Hypothesis generation for any topic
    - Statistical data and key references
    - Detailed research outline for your chosen hypothesis
    
    **Tab 4 - Text Evaluation:**
    - Comprehensive grading (0-10) for spelling, grammar, style, coherence
    - Text type detection and analysis
    - Specific corrections and recommendations
    
    **Tab 5 - LaTeX Maker:**
    - Converts text to LaTeX code for RMarkdown
    - Multiple document types supported
    - Ready-to-use code with proper formatting
    
    **Tab 6 - Reference Maker:**
    - Creates properly formatted references
    - Supports APA, MLA, and Chicago styles
    - Multiple reference types (articles, books, websites, etc.)
    """)
    
    # Function to get corrections for transformed text
    def get_tone_corrections(text, model_id):
        corrections_prompt = f"""
        Analyze the following text for corrections:
        
        Text:
        {text}
        
        Provide analysis in this format:
        
        COHERENCE: [Analysis and suggestions]
        STYLE: [Analysis and suggestions]
        GRAMMAR: [Analysis and suggestions]
        OTHER CORRECTIONS: [Any other improvements needed]
        """
        
        return call_bedrock_model(corrections_prompt, model_id)
    
    # Generate tone change button
    if st.button("Transform Text Tone", key="change_tone"):
        if not tone_text.strip():
            st.error("Please enter some text to transform.")
        elif not MODEL_ID:
            st.error("⚠️ Please select a model in the sidebar first")
        else:
            with st.spinner(f"Transforming text to {selected_tone.lower()} tone..."):
                tone_results = {}
                sections = [
                    "hypothesis", "main_bullet_points", "most_important_data_points", 
                    "summary", "abstract", "introduction", "body_text", "conclusion", "appendix"
                ]
                
                for section in sections:
                    tone_results[section] = change_text_tone_section_enhanced(
                        tone_text, selected_tone, section, selected_text_type,
                        technical_level, formality_level, statistics_level, MODEL_ID
                    )
                
                # Get corrections for the transformed text
                corrections = get_tone_corrections(tone_text, MODEL_ID)
                tone_results["corrections"] = corrections
                
                st.session_state["tone_results"] = tone_results
                st.session_state["current_tone"] = selected_tone
                st.session_state["current_settings"] = {
                    "tone": selected_tone,
                    "text_type": selected_text_type,
                    "technical_level": technical_level,
                    "formality_level": formality_level,
                    "statistics_level": statistics_level
                }
    
    # Display results with regenerate buttons
    if "tone_results" in st.session_state:
        st.subheader(f"Text Transformed - Settings Applied")
        st.write(f"**Style:** {st.session_state['current_settings']['tone']} | "
                f"**Type:** {st.session_state['current_settings']['text_type']} | "
                f"**Technical:** {st.session_state['current_settings']['technical_level']} | "
                f"**Formality:** {st.session_state['current_settings']['formality_level']} | "
                f"**Statistics:** {st.session_state['current_settings']['statistics_level']}")
        
        sections = [
            ("hypothesis", "Hypothesis"),
            ("main_bullet_points", "Main Bullet Points"),
            ("most_important_data_points", "Most Important Data Points"),
            ("summary", "Summary"),
            ("abstract", "Abstract"),
            ("introduction", "Introduction"),
            ("body_text", "Body Text"),
            ("conclusion", "Conclusion"),
            ("appendix", "Appendix")
        ]
        
        for key, title in sections:
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.text_area(
                    title,
                    value=st.session_state["tone_results"].get(key, ""),
                    height=150,
                    key=f"tone_result_{key}"
                )
            
            with col2:
                if st.button(f"Regenerate", key=f"regen_{key}"):
                    if MODEL_ID and tone_text.strip():
                        with st.spinner(f"Regenerating {title.lower()}..."):
                            settings = st.session_state.get('current_settings', {})
                            new_result = change_text_tone_section_enhanced(
                                tone_text,
                                settings.get('tone', selected_tone),
                                key,
                                settings.get('text_type', selected_text_type),
                                settings.get('technical_level', technical_level),
                                settings.get('formality_level', formality_level),
                                settings.get('statistics_level', statistics_level),
                                MODEL_ID
                            )
                            st.session_state["tone_results"][key] = new_result
                            st.rerun()
        
        # Display corrections
        if "corrections" in st.session_state["tone_results"]:
            st.subheader("Text Analysis and Corrections")
            st.text_area(
                "Coherence, Style, Grammar and Other Corrections:",
                value=st.session_state["tone_results"]["corrections"],
                height=200,
                key="tone_corrections_display"
            )

# ==== TAB 3: TOPIC IDEA EXPLORER (ENHANCED) ====
with tab3:
    st.header("Topic Idea Explorer")
    st.write("Explore hypothesis ideas for any topic and get relevant statistics, references, and proposed outlines.")
    
    # Topic input
    topic_input = st.text_input("Enter a topic you want to explore:", key="topic_input")
    
    # Function to generate hypothesis options
    def generate_hypothesis_options(topic, model_id):
        prompt = f"""
        Generate 10 different research hypothesis options for the topic: {topic}
        
        Each hypothesis should be:
        - Specific and testable
        - Relevant to the topic
        - Academically sound
        - Numbered from 1 to 10
        
        Format as:
        1. [First hypothesis]
        2. [Second hypothesis]
        ...
        10. [Tenth hypothesis]
        """
        
        return call_bedrock_model(prompt, model_id)
    
    # Function to get statistics for selected hypothesis
    def get_hypothesis_statistics(hypothesis, model_id):
        prompt = f"""
        Provide main statistics and data points related to this hypothesis: {hypothesis}
        
        Include:
        - Relevant numerical data
        - Key statistics
        - Important metrics
        - Sample sizes or populations when relevant
        - Any significant findings from existing research
        
        Present this information in a clear, organized manner.
        """
        
        return call_bedrock_model(prompt, model_id)
    
    # Function to get references for selected hypothesis
    def get_hypothesis_references(hypothesis, model_id):
        prompt = f"""
        Provide the most important academic references and sources that a researcher should check 
        for this hypothesis: {hypothesis}
        
        Include:
        - Key academic papers or studies
        - Important books on the topic
        - Relevant journals
        - Government or institutional reports
        - Online databases or resources
        
        Format as a list with brief descriptions of why each source is important.
        """
        
        return call_bedrock_model(prompt, model_id)
    
    # Function to get proposed outline
    def get_hypothesis_outline(hypothesis, model_id):
        prompt = f"""
        Create a detailed proposed outline for a research text based on this hypothesis: {hypothesis}
        
        The outline should include:
        - Introduction section with subsections
        - Literature review structure
        - Methodology section
        - Results/Analysis section
        - Discussion section
        - Conclusion section
        - References section
        
        Format as a hierarchical outline with main sections and subsections.
        Make it detailed enough that a researcher can use it as a framework to write their paper.
        """
        
        return call_bedrock_model(prompt, model_id)
    
    # Generate hypotheses button
    if st.button("Generate Hypothesis Options", key="generate_hypotheses"):
        if not topic_input.strip():
            st.error("Please enter a topic to explore.")
        elif not MODEL_ID:
            st.error("⚠️ Please select a model in the sidebar first")
        else:
            with st.spinner("Generating hypothesis options..."):
                hypotheses = generate_hypothesis_options(topic_input, MODEL_ID)
                st.session_state["hypotheses"] = hypotheses
                st.session_state["selected_hypothesis"] = None
    
    # Display hypotheses if available
    if "hypotheses" in st.session_state:
        st.subheader("Generated Hypothesis Options")
        st.text_area("Choose from these hypotheses:", value=st.session_state["hypotheses"], height=300, key="hypotheses_display")
        
        # Input for selected hypothesis
        selected_hypothesis = st.text_area(
            "Copy and paste the hypothesis you want to explore further:",
            height=100,
            key="selected_hypothesis_input"
        )
        
        # Button to get statistics, references, and outline
        if st.button("Get Statistics, References & Outline", key="get_stats_refs_outline"):
            if not selected_hypothesis.strip():
                st.error("Please select a hypothesis first.")
            else:
                with st.spinner("Getting statistics, references, and outline..."):
                    # Get statistics
                    statistics = get_hypothesis_statistics(selected_hypothesis, MODEL_ID)
                    # Get references
                    references = get_hypothesis_references(selected_hypothesis, MODEL_ID)
                    # Get outline
                    outline = get_hypothesis_outline(selected_hypothesis, MODEL_ID)
                    
                    st.session_state["hypothesis_statistics"] = statistics
                    st.session_state["hypothesis_references"] = references
                    st.session_state["hypothesis_outline"] = outline
        
        # Display statistics, references, and outline
        if "hypothesis_statistics" in st.session_state:
            # First row: Statistics and References
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Main Statistics")
                st.text_area(
                    "Key statistical data for your hypothesis:",
                    value=st.session_state["hypothesis_statistics"],
                    height=300,
                    key="stats_display"
                )
            
            with col2:
                st.subheader("Important References")
                st.text_area(
                    "Key sources to check for your hypothesis:",
                    value=st.session_state["hypothesis_references"],
                    height=300,
                    key="refs_display"
                )
            
            # Second row: Proposed Outline
            if "hypothesis_outline" in st.session_state:
                st.subheader("Proposed Research Outline")
                st.text_area(
                    "Detailed outline for your research text:",
                    value=st.session_state["hypothesis_outline"],
                    height=400,
                    key="outline_display"
                )

# ==== TAB 4: TEXT EVALUATION ====
with tab4:
    st.header("Text Evaluation")
    st.write("Get a comprehensive evaluation of your text with grades and specific corrections.")
    
    # Input text area
    evaluation_text = st.text_area("Enter your text for evaluation:", height=200, key="evaluation_text_input")
    
    # Function to evaluate text
    def evaluate_text_comprehensive(text, model_id):
        prompt = f"""
        Evaluate the following text comprehensively across multiple dimensions. Provide grades from 0 to 10 and specific corrections where needed.
        
        Text to evaluate:
        {text}
        
        Please provide your evaluation in this exact format:
        
        SPELLING EVALUATION:
        Grade: [0-10]
        Corrections: [List spelling errors and corrections, or "No spelling errors found"]
        
        GRAMMAR EVALUATION:
        Grade: [0-10]
        Corrections: [List grammar errors and corrections, or "No grammar errors found"]
        
        STYLE EVALUATION:
        Grade: [0-10]
        Text Type Detected: [e.g., Academic paper, Report, Blog post, etc.]
        Style Analysis: [Analysis of writing style and suggestions for improvement]
        
        COHERENCE EVALUATION:
        Grade: [0-10]
        Corrections: [List coherence issues and suggestions, or "Text is coherent"]
        
        OVERALL EVALUATION:
        Grade: [0-10]
        Overall Corrections: [Summary of main issues and recommendations for improvement]
        """
        
        return call_bedrock_model(prompt, model_id)
    
    # Function to parse evaluation results
    def parse_evaluation_results(evaluation_text):
        sections = {}
        
        # Define the sections to extract
        section_markers = [
            ("SPELLING EVALUATION:", "spelling"),
            ("GRAMMAR EVALUATION:", "grammar"),
            ("STYLE EVALUATION:", "style"),
            ("COHERENCE EVALUATION:", "coherence"),
            ("OVERALL EVALUATION:", "overall")
        ]
        
        for marker, key in section_markers:
            if marker in evaluation_text:
                start_idx = evaluation_text.find(marker) + len(marker)
                # Find the next section or end of text
                next_markers = [m for m, _ in section_markers if m != marker and m in evaluation_text]
                if next_markers:
                    next_indices = [evaluation_text.find(m) for m in next_markers if evaluation_text.find(m) > start_idx]
                    if next_indices:
                        next_idx = min(next_indices)
                        sections[key] = evaluation_text[start_idx:next_idx].strip()
                    else:
                        sections[key] = evaluation_text[start_idx:].strip()
                else:
                    sections[key] = evaluation_text[start_idx:].strip()
        
        return sections

    # Evaluate text button
    if st.button("Evaluate Text", key="evaluate_text"):
        if not evaluation_text.strip():
            st.error("Please enter some text to evaluate.")
        elif not MODEL_ID:
            st.error("⚠️ Please select a model in the sidebar first")
        else:
            with st.spinner("Evaluating your text..."):
                evaluation_results = evaluate_text_comprehensive(evaluation_text, MODEL_ID)
                parsed_results = parse_evaluation_results(evaluation_results)
                st.session_state["evaluation_results"] = parsed_results
                st.session_state["raw_evaluation"] = evaluation_results

    # Display evaluation results
    if "evaluation_results" in st.session_state:
        st.subheader("Text Evaluation Results")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        evaluation_sections = [
            ("spelling", "Spelling Evaluation"),
            ("grammar", "Grammar Evaluation"),
            ("style", "Style Evaluation"),
            ("coherence", "Coherence Evaluation"),
            ("overall", "Overall Evaluation")
        ]
        
        for i, (key, title) in enumerate(evaluation_sections):
            with col1 if i % 2 == 0 else col2:
                if key in st.session_state["evaluation_results"]:
                    st.text_area(
                        title,
                        value=st.session_state["evaluation_results"][key],
                        height=200,
                        key=f"eval_{key}"
                    )
                else:
                    st.text_area(
                        title,
                        value="No evaluation available for this section.",
                        height=200,
                        key=f"eval_{key}_empty"
                    )

# ==== TAB 5: LATEX MAKER ====
with tab5:
    st.header("LaTeX Maker")
    st.write("Convert your text into properly formatted LaTeX code for RMarkdown that can be knitted into a PDF.")
    
    # Text type selection for LaTeX
    latex_text_type = st.selectbox(
        "Select the type of document:",
        ["Academic Paper", "Report", "Press Release", "Technical Manual", "Thesis", "Article"],
        key="latex_text_type"
    )
    
    # Input text area
    latex_text = st.text_area("Enter your text to convert to LaTeX:", height=200, key="latex_text_input")
    
    # Function to generate LaTeX code
    def generate_latex_code(text, document_type, model_id):
        prompt = f"""
        Convert the following text into properly formatted LaTeX code suitable for RMarkdown that can be knitted into a PDF.
        
        Document Type: {document_type}
        
        Text to convert:
        {text}
        
        Requirements:
        1. Create a complete LaTeX document structure appropriate for a {document_type.lower()}
        2. Include proper document class and packages
        3. Format any equations, numbers, or special formatting appropriately
        4. Add proper sectioning (\\section, \\subsection, etc.)
        5. Include title, author, date fields that can be customized
        6. Use proper LaTeX formatting for lists, emphasis, etc.
        7. Add comments explaining key formatting choices
        8. Make it compatible with RMarkdown output: pdf_document
        
        Provide only the LaTeX code that can be copied and pasted into RMarkdown.
        """
        
        return call_bedrock_model(prompt, model_id)
    
    # Generate LaTeX button
    if st.button("Generate LaTeX Code", key="generate_latex"):
        if not latex_text.strip():
            st.error("Please enter some text to convert.")
        elif not MODEL_ID:
            st.error("⚠️ Please select a model in the sidebar first")
        else:
            with st.spinner("Generating LaTeX code..."):
                latex_code = generate_latex_code(latex_text, latex_text_type, MODEL_ID)
                st.session_state["latex_code"] = latex_code
    
    # Display LaTeX code
    if "latex_code" in st.session_state:
        st.subheader("Generated LaTeX Code")
        st.text_area(
            "Copy this code into your RMarkdown document:",
            value=st.session_state["latex_code"],
            height=500,
            key="latex_output"
        )
        
        # Instructions for using the code
        st.info("""
        **How to use this LaTeX code:**
        1. Copy the generated code above
        2. Create a new RMarkdown file (.Rmd)
        3. Set the output to `pdf_document` in the YAML header
        4. Paste the LaTeX code in the document body
        5. Knit to PDF
        
        **Sample YAML header:**
        ```yaml
        ---
        title: "Your Title"
        author: "Your Name"
        date: "`r Sys.Date()`"
        output: pdf_document
        ---
        ```
        """)

# ==== TAB 6: REFERENCE MAKER ====
with tab6:
    st.header("Reference Maker")
    st.write("Create properly formatted references in APA, MLA, or Chicago style.")
    
    # Reference style selection
    reference_style = st.selectbox(
        "Select reference style:",
        ["APA", "MLA", "Chicago"],
        key="reference_style"
    )
    
    # Reference type selection
    reference_type = st.selectbox(
        "Select reference type:",
        ["Journal Article", "Book", "Website", "Conference Paper", "Thesis/Dissertation", "Report"],
        key="reference_type"
    )
    
    # Create input fields based on reference type
    col1, col2 = st.columns(2)
    
    with col1:
        author = st.text_input("Author(s):", key="ref_author")
        title = st.text_input("Title:", key="ref_title")
        year = st.text_input("Year Published:", key="ref_year")
        
        if reference_type in ["Journal Article", "Conference Paper"]:
            journal_conference = st.text_input("Journal/Conference Name:", key="ref_journal")
            volume = st.text_input("Volume:", key="ref_volume")
            issue = st.text_input("Issue:", key="ref_issue")
            pages = st.text_input("Pages:", key="ref_pages")
        
        if reference_type == "Book":
            publisher = st.text_input("Publisher:", key="ref_publisher")
            place_published = st.text_input("Place Published:", key="ref_place")
            edition = st.text_input("Edition (if not first):", key="ref_edition")
        
        if reference_type == "Website":
            website_name = st.text_input("Website Name:", key="ref_website")
            url = st.text_input("URL:", key="ref_url")
            access_date = st.text_input("Date Accessed:", key="ref_access_date")
    
    with col2:
        if reference_type == "Thesis/Dissertation":
            institution = st.text_input("Institution:", key="ref_institution")
            degree_type = st.text_input("Degree Type (Master's/PhD):", key="ref_degree")
            department = st.text_input("Department:", key="ref_department")
        
        if reference_type == "Report":
            organization = st.text_input("Organization:", key="ref_organization")
            report_number = st.text_input("Report Number:", key="ref_report_num")
            place_published = st.text_input("Place Published:", key="ref_report_place")
        
        # Additional fields for all types
        doi = st.text_input("DOI (if available):", key="ref_doi")
        notes = st.text_area("Additional Notes:", key="ref_notes", height=100)
    
    # Function to generate reference
    def generate_reference(style, ref_type, fields, model_id):
        # Create a string with all the provided information
        field_info = "\n".join([f"{key}: {value}" for key, value in fields.items() if value.strip()])
        
        prompt = f"""
        Create a properly formatted reference in {style} style for a {ref_type.lower()}.
        
        Reference Information:
        {field_info}
        
        Requirements:
        1. Follow {style} formatting guidelines exactly
        2. Include all provided information in the correct order
        3. Use proper punctuation, italics, and formatting
        4. Include DOI if provided
        5. Handle missing information appropriately
        6. Provide only the formatted reference
        
        Format the reference exactly as it should appear in a reference list.
        """
        
        return call_bedrock_model(prompt, model_id)
    
    # Generate reference button
    if st.button("Generate Reference", key="generate_reference"):
        if not author.strip() or not title.strip():
            st.error("Please provide at least author and title information.")
        elif not MODEL_ID:
            st.error("⚠️ Please select a model in the sidebar first")
        else:
            # Collect all field information
            fields = {
                "Author(s)": author,
                "Title": title,
                "Year": year,
                "DOI": doi,
                "Notes": notes
            }
            
            # Add type-specific fields
            if reference_type in ["Journal Article", "Conference Paper"]:
                fields["Journal/Conference"] = journal_conference
                fields["Volume"] = volume
                fields["Issue"] = issue
                fields["Pages"] = pages
            elif reference_type == "Book":
                fields["Publisher"] = publisher
                fields["Place Published"] = place_published
                fields["Edition"] = edition
            elif reference_type == "Website":
                fields["Website Name"] = website_name
                fields["URL"] = url
                fields["Date Accessed"] = access_date
            elif reference_type == "Thesis/Dissertation":
                fields["Institution"] = institution
                fields["Degree Type"] = degree_type
                fields["Department"] = department
            elif reference_type == "Report":
                fields["Organization"] = organization
                fields["Report Number"] = report_number
                fields["Place Published"] = place_published
            
            with st.spinner("Generating reference..."):
                formatted_reference = generate_reference(reference_style, reference_type, fields, MODEL_ID)
                st.session_state["formatted_reference"] = formatted_reference
    
    # Display generated reference
    if "formatted_reference" in st.session_state:
        st.subheader("Generated Reference")
        st.text_area(
            f"Copy this {reference_style} style reference:",
            value=st.session_state["formatted_reference"],
            height=150,
            key="reference_output"
        )
        
        st.success("✅ Reference generated successfully! You can copy and paste it into your document.")

# ==== TAB 7: SETUP HELP ====
with tab7:
    st.header("AWS Bedrock Setup Guide")
    
    st.markdown("""
    ## Troubleshooting AWS Bedrock Access
    
    If you're getting "AccessDeniedException" errors, follow these steps:
    
    ### 1. Verify AWS Region
    
    Bedrock is only available in specific regions:
    - US East (N. Virginia): `us-east-1`
    - US West (Oregon): `us-west-2`
    - Europe (Frankfurt): `eu-central-1`
    - Asia Pacific (Tokyo): `ap-northeast-1`
    
    Make sure you select one of these regions in the sidebar.
    
    ### 2. Enable Bedrock Model Access
    
    You need to request access to the specific models you want to use:
    
    1. Go to the [AWS Bedrock console](https://console.aws.amazon.com/bedrock/)
    2. Click on "Model access" in the left sidebar
    3. Click "Manage model access"
    4. Select the models you want to use (Claude, Titan, etc.)
    5. Click "Request model access"
    6. Wait for approval (some models are approved instantly)
    
    ### 3. Check IAM Permissions
    
    Ensure your IAM user/role has these permissions:
    
    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:ListFoundationModels",
                    "bedrock:GetFoundationModel"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock-runtime:InvokeModel"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucket",
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:CreateBucket",
                    "s3:ListAllMyBuckets"
                ],
                "Resource": [
                    "arn:aws:s3:::*",
                    "arn:aws:s3:::*/*"
                ]
            }
        ]
    }
    ```
    
    ### 4. Configure AWS CLI
    
    Make sure your AWS CLI is properly configured:
    
    ```bash
    aws configure --profile recruitment-assistant
    ```
    
    Enter your AWS Access Key ID, Secret Access Key, region (e.g., us-east-1), and output format (json).
    
    ### 5. Test Your Access
    
    Use these commands to test if you can list Bedrock models:
    
    ```bash
    aws bedrock list-foundation-models --region us-east-1 --profile recruitment-assistant
    ```
    
    Test S3:
    
    ```bash
    aws s3 ls --profile recruitment-assistant
    ```
    
    If these work, you should be able to access these services through this app.
    
    ### 6. Application Features
    
    This enhanced version includes:
    
    **Tab 1 - User Own Text:**
    - Comprehensive text analysis with sections detection
    - Detailed corrections for spelling, grammar, coherence, style, and order
    - Proposed corrected version of your text
    
    **Tab 2 - Change Text Tone:**
    - Advanced tone transformation with customizable parameters
    - Technical level, formality, and statistics usage controls
    - Text type selection (report, academic paper, etc.)
    - Coherence and style analysis of transformed text
    
    **Tab 3 - Topic Idea Explorer:**
    - Hypothesis generation for any topic
    - Statistical data and key references
    - Detailed research outline for your chosen hypothesis
    
    **Tab 4 - Text Evaluation:**
    - Comprehensive grading (0-10) for spelling, grammar, style, coherence
    - Text type detection and analysis
    - Specific corrections and recommendations
    
    **Tab 5 - LaTeX Maker:**
    - Converts text to LaTeX code for RMarkdown
    - Multiple document types supported
    - Ready-to-use code with proper formatting
    
    **Tab 6 - Reference Maker:**
    - Creates properly formatted references
    - Supports APA, MLA, and Chicago styles
    - Multiple reference types (articles, books, websites, etc.)
    """)

# ===== MAIN APP FOOTER =====
st.sidebar.markdown("---")
st.sidebar.markdown("**AI-Powered Text Analysis Assistant v2.0**")
st.sidebar.markdown("Enhanced with comprehensive analysis features")

# Display connection status
if bedrock_runtime and MODEL_ID:
    st.sidebar.success(f"✅ Connected to {MODEL_ID}")
else:
    st.sidebar.warning("⚠️ Not connected to Bedrock")

if s3 and S3_BUCKET:
    st.sidebar.success(f"✅ S3 Bucket: {S3_BUCKET}")
else:
    st.sidebar.warning("⚠️ S3 not configured")