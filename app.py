import re
import json
import pandas as pd
import spacy
from spacy.matcher import Matcher
from word2number import w2n
from flask import Flask, request, jsonify

# Load your Medicine.csv file (200-row dataset) in same folder:
df = pd.read_csv("Medicine.csv")
price_map = dict(zip(df.drug.str.lower(), df.unit_price))

def word_to_int(tok: str) -> int:
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return w2n.word_to_num(tok)

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Patterns
matcher.add("DRUG", [[{"LOWER": {"IN": list(price_map.keys())}}]])
matcher.add("FREQ", [[
    {"TEXT": {"REGEX": "^\\d+$|^[a-zA-Z]+$"}},
    {"LOWER": {"IN": ["per", "times"]}, "OP": "?"},
    {"LOWER": {"IN": ["day", "daily"]}, "OP": "+"}
]])
matcher.add("DUR", [[
    {"TEXT": {"REGEX": "^\\d+$|^[a-zA-Z]+$"}},
    {"LOWER": {"IN": ["day", "days"]}}
]])

def parse_and_calculate(text: str) -> dict:
    txt = text.lower()
    doc = nlp(txt)
    matches = matcher(doc)
    res = {"drug": None, "frequency": None, "duration": None}

    for mid, start, end in matches:
        label = nlp.vocab.strings[mid]
        span = doc[start:end].text
        if label == "DRUG":
            res["drug"] = span
        else:
            try:
                num = word_to_int(span)
            except:
                continue
            if label == "FREQ":
                res["frequency"] = num
            elif label == "DUR":
                res["duration"] = num

    # Fallback regex if needed
    if res["drug"] and (res["frequency"] is None or res["duration"] is None):
        d = re.escape(res["drug"])
        pat = rf"{d}\s+(?P<freq>\d+|[A-Za-z]+)\s*(?:per|times)?\s*(?:day|daily)\s*(?:for)?\s*(?P<dur>\d+|[A-Za-z]+)\s*(?:day|days)"
        m = re.search(pat, txt)
        if m:
            if res["frequency"] is None:
                try: res["frequency"] = word_to_int(m.group("freq"))
                except: pass
            if res["duration"] is None:
                try: res["duration"] = word_to_int(m.group("dur"))
                except: pass

    # Compute cost
    if res["drug"] and res["frequency"] and res["duration"]:
        total = res["frequency"] * res["duration"]
        up = price_map.get(res["drug"], 0.0)
        res.update({
            "total_units": total,
            "unit_price": up,
            "cost": round(total * up, 2)
        })
    return res

app = Flask(__name__)

@app.route("/calculate_bill", methods=["POST"])
def calculate_bill():
    data = request.get_json(force=True)
    presc = data.get("prescription", "")
    return jsonify(parse_and_calculate(presc))

if __name__ == "__main__":
    app.run(debug=True)
