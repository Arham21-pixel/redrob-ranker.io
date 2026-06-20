import os
import json
import re
import docx

def parse_jd(docx_path: str, output_path: str):
    """
    Reads the job_description.docx file and extracts structured information 
    such as skills, rules, and preferences based on the document structure.
    """
    doc = docx.Document(docx_path)
    # Join paragraphs, keeping only non-empty lines
    text = '\n'.join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    
    # 1. Experience range
    exp_match = re.search(r"What we mean by \"(\d+)-(\d+) years\"", text)
    if exp_match:
        exp_min = int(exp_match.group(1))
        exp_max = int(exp_match.group(2))
    else:
        exp_min, exp_max = 5, 9
        
    # 2. Location preferences (Hardcoded from 'Location: Pune/Noida-preferred...' section)
    # Explicit list is safer for exact string matching downstream.
    locations = ["Pune", "Noida", "Hyderabad", "Mumbai", "Delhi NCR"]
    
    # 3. Notice period preference
    notice_match = re.search(r"sub-(\d+)-day notice", text)
    notice_period = int(notice_match.group(1)) if notice_match else 30
    
    # 4. Required Skills
    req_skills = []
    try:
        req_skills_section = text.split("Things you absolutely need")[1].split("Things we'd like you to have")[0]
        req_skills = [s.strip() for s in req_skills_section.strip().split('\n') if s.strip()]
    except Exception as e:
        print("Warning: Could not parse required skills section.")
        
    # 5. Nice to have Skills
    nice_skills = []
    try:
        nice_skills_section = text.split("Things we'd like you to have but won't reject you for")[1].split("Things we explicitly do NOT want")[0]
        nice_skills = [s.strip() for s in nice_skills_section.strip().split('\n') if s.strip()]
    except Exception as e:
        print("Warning: Could not parse nice-to-have skills section.")
        
    # 6. Disqualifier Rules
    disq_1 = []
    try:
        disq_section_1 = text.split("disqualifiers we actually apply:")[1].split("The skills inventory")[0]
        disq_1 = [s.strip() for s in disq_section_1.strip().split('\n') if s.strip()]
    except Exception as e:
        pass
        
    disq_2 = []
    try:
        disq_section_2 = text.split("Things we explicitly do NOT want")[1].split("This is the section most JDs skip but we think it's the most important:")[1].split("On location, comp, and logistics")[0]
        disq_2 = [s.strip() for s in disq_section_2.strip().split('\n') if s.strip()]
    except Exception as e:
        pass
        
    disqualifier_rules = disq_1 + disq_2
    
    # 7. Ideal Profile Signals
    ideal_profile_signals = ""
    try:
        ideal_profile = text.split("The \"ideal candidate\" we're imagining is roughly:")[1].split("We are aware this is a narrow profile.")[0]
        # Replace newlines with spaces to make it a continuous text description
        ideal_profile_signals = ' '.join([s.strip() for s in ideal_profile.strip().split('\n') if s.strip()])
    except Exception as e:
        print("Warning: Could not parse ideal profile signals section.")

    # Combine into schema
    parsed_data = {
        "required_skills": req_skills,
        "nice_to_have_skills": nice_skills,
        "disqualifier_rules": disqualifier_rules,
        "ideal_profile_signals": ideal_profile_signals,
        "location_preference": locations,
        "notice_period_preference_days": notice_period,
        "experience_range_years": [exp_min, exp_max]
    }
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=4)
        
    print(f"Successfully parsed JD text to {output_path}")

if __name__ == "__main__":
    input_jd = os.path.join("data", "raw", "job_description.docx")
    output_json = os.path.join("data", "processed", "jd_parsed.json")
    parse_jd(input_jd, output_json)
