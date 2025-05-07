import pandas as pd
import spacy
from spacy.matcher import Matcher
from word2number import w2n
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load Medicine.csv and create price_map with error handling
try:
    df = pd.read_csv("Medicine.csv")
except FileNotFoundError:
    print("Error: Medicine.csv file not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print("Error: Medicine.csv file is empty.")
    exit(1)
except pd.errors.ParserError:
    print("Error: Medicine.csv file has parsing errors.")
    exit(1)

price_map = dict(zip(df.drug.str.lower(), df.unit_price))

# Initialize spaCy and Matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define patterns for drugs, frequencies, durations, and meal instructions
matcher.add("DRUG", [[{"LOWER": {"IN": list(price_map.keys())}}]])
freq_patterns = [
    [{"LIKE_NUM": True}, {"LOWER": "times"}, {"LOWER": "a", "OP": "?"}, {"LOWER": "day"}],
    [{"LIKE_NUM": True}, {"LOWER": "per"}, {"LOWER": "day"}],
    [{"LIKE_NUM": True}, {"LOWER": "daily"}],
    [{"LOWER": {"IN": ["once", "twice", "thrice"]}}, {"LOWER": "a", "OP": "?"}, {"LOWER": "day"}],
    [{"LOWER": {"IN": ["once", "twice", "thrice"]}}, {"LOWER": "daily"}]
]
matcher.add("FREQ", freq_patterns)
dur_patterns = [
    [{"LOWER": "for"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["day", "days"]}}]
]
matcher.add("DUR", dur_patterns)
meal_patterns = [
    [{"LOWER": {"IN": ["before", "after"]}}, {"LOWER": {"IN": ["meal", "meals"]}}],
    [{"LOWER": {"IN": ["before", "after"]}}, {"LOWER": {"IN": ["breakfast", "lunch", "dinner"]}}]
]
matcher.add("MEAL", meal_patterns)

def word_to_int(tok: str) -> int:
    """Convert a token to an integer, handling special cases like 'once', 'twice', 'thrice'."""
    tok = tok.lower()
    if tok in ["once", "one"]:
        return 1
    elif tok in ["twice", "two"]:
        return 2
    elif tok in ["thrice", "three"]:
        return 3
    elif tok.isdigit():
        return int(tok)
    else:
        try:
            return w2n.word_to_num(tok)
        except ValueError:
            return None

def extract_number(span):
    """Extract the number from a span for frequency or duration."""
    for token in span:
        num = word_to_int(token.text)
        if num is not None:
            return num
    return None

def parse_and_calculate(text: str) -> dict:
    """Parse prescription text and calculate total cost using prices from Medicine.csv."""
    txt = text.lower()
    doc = nlp(txt)
    matches = matcher(doc)
    matches = sorted(matches, key=lambda x: x[1])  # Sort by start position
    
    results = []
    current_drug = None
    
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = doc[start:end]
        if label == "DRUG":
            if current_drug is not None:
                # If duration is not specified, assume 1 day
                if current_drug["duration"] is None:
                    current_drug["duration"] = 1
                results.append(current_drug)
            current_drug = {"drug": span.text, "frequency": None, "duration": None, "meal_instruction": None}
        elif label == "FREQ" and current_drug is not None:
            if current_drug["frequency"] is None:
                num = extract_number(span)
                if num is not None:
                    current_drug["frequency"] = num
        elif label == "DUR" and current_drug is not None:
            if current_drug["duration"] is None:
                num = extract_number(span)
                if num is not None:
                    current_drug["duration"] = num
        elif label == "MEAL" and current_drug is not None:
            if current_drug["meal_instruction"] is None:
                current_drug["meal_instruction"] = span.text
    
    if current_drug is not None:
        # If duration is not specified, assume 1 day
        if current_drug["duration"] is None:
            current_drug["duration"] = 1
        results.append(current_drug)
    
    # Calculate costs using price_map from Medicine.csv
    total_cost = 0.0
    for res in results:
        if res["drug"] and res["frequency"] and res["duration"]:
            total = res["frequency"] * res["duration"]
            unit_price = price_map.get(res["drug"], 0.0)
            cost = round(total * unit_price, 2)
            res.update({
                "total_units": total,
                "unit_price": unit_price,
                "cost": cost
            })
            total_cost += cost
        else:
            res.update({
                "total_units": None,
                "unit_price": None,
                "cost": None
            })
    
    return {"medicines": results, "total_cost": total_cost}

@app.route('/calculate_bill', methods=['POST'])
def calculate_bill():
    """Flask endpoint to calculate bill from prescription text."""
    data = request.get_json(force=True)
    prescription = data.get('prescription', '')
    result = parse_and_calculate(prescription)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)