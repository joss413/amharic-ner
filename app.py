import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import gradio as gr

# ── Load model ────────────────────────────────────────────
with open('saved_model/ner_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

id2label    = {int(k): v for k, v in config['id2label'].items()}
label2id    = config['label2id']
label_names = config['label_names']

tokenizer = AutoTokenizer.from_pretrained('saved_model/')
model     = AutoModelForTokenClassification.from_pretrained('saved_model/')
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = model.to(device)
model.eval()

# ── Prediction ────────────────────────────────────────────
def predict_ner(text):
    if not text.strip():
        return "<p class='empty'>Please enter some Amharic text — የአማርኛ ጽሑፍ ያስገቡ</p>"

    words     = text.strip().split()
    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors='pt',
        max_length=128,
        padding='max_length',
        truncation=True
    )

    input_ids      = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    predictions      = torch.argmax(outputs.logits, dim=-1)[0]
    word_ids         = tokenized.word_ids()
    word_predictions = {}

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id not in word_predictions:
            word_predictions[word_id] = id2label[predictions[idx].item()]

    entity_colors = {
        'PER':  {'bg': '#e53e3e', 'light': 'rgba(229,62,62,0.12)', 'border': '#fc8181', 'glow': 'rgba(229,62,62,0.3)'},
        'LOC':  {'bg': '#38a169', 'light': 'rgba(56,161,105,0.12)', 'border': '#68d391', 'glow': 'rgba(56,161,105,0.3)'},
        'ORG':  {'bg': '#3182ce', 'light': 'rgba(49,130,206,0.12)', 'border': '#63b3ed', 'glow': 'rgba(49,130,206,0.3)'},
        'DATE': {'bg': '#d69e2e', 'light': 'rgba(214,158,46,0.12)', 'border': '#f6e05e', 'glow': 'rgba(214,158,46,0.3)'},
    }

    tag_map = {
        'B-PER': 'PER', 'I-PER': 'PER',
        'B-LOC': 'LOC', 'I-LOC': 'LOC',
        'B-ORG': 'ORG', 'I-ORG': 'ORG',
        'B-DATE': 'DATE', 'I-DATE': 'DATE',
    }

    entity_counts = {'PER': 0, 'LOC': 0, 'ORG': 0, 'DATE': 0}
    prev_tag = None
    for word_id in range(len(words)):
        tag = word_predictions.get(word_id, 'O')
        entity = tag_map.get(tag)
        if entity and (tag.startswith('B-') or prev_tag != tag):
            entity_counts[entity] += 1
        prev_tag = tag

    tokens_html = "<div class='token-row'>"
    for word_id, word in enumerate(words):
        tag    = word_predictions.get(word_id, 'O')
        entity = tag_map.get(tag)
        if entity:
            c = entity_colors[entity]
            tokens_html += f"""
            <span class='entity-chip' style='
                background: linear-gradient(135deg, {c["bg"]}dd, {c["bg"]}aa);
                border: 1.5px solid {c["border"]};
                box-shadow: 0 0 12px {c["glow"]}, 0 2px 8px rgba(0,0,0,0.3);
                color: white;
            '>
                {word}
                <span class='entity-label' style='background:rgba(255,255,255,0.2)'>{entity}</span>
            </span>"""
        else:
            tokens_html += f"<span class='plain-word'>{word}</span>"
    tokens_html += "</div>"

    summary_html = "<div class='summary-row'>"
    for entity, count in entity_counts.items():
        if count > 0:
            c = entity_colors[entity]
            label_am = {'PER': 'ሰው', 'LOC': 'ቦታ', 'ORG': 'ድርጅት', 'DATE': 'ቀን'}[entity]
            summary_html += f"""
            <div class='summary-card' style='
                background: linear-gradient(135deg, {c["light"]}, rgba(15,17,23,0.8));
                border: 1px solid {c["border"]}55;
                box-shadow: 0 4px 20px {c["glow"]};
            '>
                <div class='card-icon' style='color:{c["bg"]};text-shadow:0 0 20px {c["glow"]}'>
                    {"👤" if entity=="PER" else "📍" if entity=="LOC" else "🏢" if entity=="ORG" else "📅"}
                </div>
                <div class='card-count' style='color:{c["bg"]};text-shadow:0 0 15px {c["glow"]}'>{count}</div>
                <div class='card-type'>{entity}</div>
                <div class='card-am'>{label_am}</div>
            </div>"""
    summary_html += "</div>"

    if all(v == 0 for v in entity_counts.values()):
        summary_html = "<p class='no-entity'>No named entities found in this text.</p>"

    return f"""
    <div class='result-wrapper'>
        <div class='section-label'>✦ Detected Text</div>
        {tokens_html}
        <div class='section-label' style='margin-top:28px'>✦ Entity Summary</div>
        {summary_html}
    </div>
    """

# ── Custom CSS ────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=Noto+Sans+Ethiopic:wght@400;600&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #080b12 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Animated background grid */
.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 36px 24px !important;
    background: 
        radial-gradient(ellipse at 20% 20%, rgba(49,130,206,0.06) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 80%, rgba(56,161,105,0.06) 0%, transparent 60%),
        radial-gradient(ellipse at 50% 50%, rgba(214,158,46,0.03) 0%, transparent 70%) !important;
    min-height: 100vh !important;
}

/* ── Header ── */
.app-header {
    text-align: center;
    margin-bottom: 40px;
    padding: 40px 32px 36px;
    background: linear-gradient(145deg, #0d1220, #111827);
    border: 1px solid #1a2235;
    border-radius: 20px;
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.03),
        0 20px 60px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.05);
    position: relative;
    overflow: hidden;
}

.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, 
        transparent, 
        #3182ce55, 
        #38a16955, 
        #d69e2e55, 
        #e53e3e55, 
        transparent);
}

.app-header::after {
    content: '';
    position: absolute;
    top: -80px; left: 50%;
    transform: translateX(-50%);
    width: 400px;
    height: 200px;
    background: radial-gradient(ellipse, rgba(49,130,206,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.app-flag {
    font-size: 52px;
    margin-bottom: 16px;
    display: block;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,0.4));
}

.app-title {
    font-size: 30px;
    font-weight: 600;
    color: #f0f4ff;
    letter-spacing: -0.8px;
    margin-bottom: 8px;
    text-shadow: 0 2px 20px rgba(49,130,206,0.3);
}

.app-subtitle {
    font-size: 14px;
    color: #4a5568;
    font-weight: 300;
    letter-spacing: 0.3px;
    margin-bottom: 16px;
}

.badge-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 4px;
}

.app-badge {
    display: inline-block;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #718096;
    font-size: 11px;
    padding: 5px 14px;
    border-radius: 20px;
    letter-spacing: 0.3px;
}

.app-badge b { color: #68d391; }
.app-badge.blue b { color: #63b3ed; }
.app-badge.gold b { color: #f6e05e; }

/* ── Input card ── */
.input-wrapper {
    background: linear-gradient(145deg, #0d1220, #111827);
    border: 1px solid #1a2235;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}

.input-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
}

textarea {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Noto Sans Ethiopic', 'IBM Plex Sans', sans-serif !important;
    font-size: 16px !important;
    padding: 16px !important;
    line-height: 1.9 !important;
    transition: all 0.25s ease !important;
    resize: vertical !important;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.2) !important;
}

textarea:focus {
    border-color: #3182ce55 !important;
    outline: none !important;
    box-shadow: 
        inset 0 2px 8px rgba(0,0,0,0.2),
        0 0 0 3px rgba(49,130,206,0.1),
        0 0 20px rgba(49,130,206,0.05) !important;
}

textarea::placeholder { color: #2d3748 !important; }

/* ── Button ── */
button.primary {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 10px !important;
    color: white !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 13px 28px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    margin-top: 14px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.3), 0 0 30px rgba(37,99,235,0.1) !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.4), 0 0 40px rgba(37,99,235,0.15) !important;
    transform: translateY(-1px) !important;
}

button.primary:active { transform: translateY(0) !important; }

/* ── Output card ── */
.output-card {
    background: linear-gradient(145deg, #0d1220, #111827) !important;
    border: 1px solid #1a2235 !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
    min-height: 80px !important;
}

/* ── Result content ── */
.result-wrapper {
    padding: 4px 0;
    font-family: 'IBM Plex Sans', sans-serif;
}

.section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2d3748;
    margin-bottom: 14px;
}

.token-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    padding: 20px;
    background: rgba(0,0,0,0.25);
    border-radius: 12px;
    border: 1px solid #1a2235;
    min-height: 66px;
    line-height: 2.6;
}

.entity-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 8px;
    font-family: 'Noto Sans Ethiopic', sans-serif;
    font-size: 15px;
    font-weight: 500;
    cursor: default;
    transition: all 0.2s ease;
    position: relative;
}

.entity-chip:hover {
    transform: translateY(-2px);
    filter: brightness(1.1);
}

.entity-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 2px 5px;
    border-radius: 4px;
}

.plain-word {
    font-family: 'Noto Sans Ethiopic', sans-serif;
    font-size: 15px;
    color: #718096;
    padding: 5px 3px;
}

/* ── Summary cards ── */
.summary-row {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin-top: 4px;
}

.summary-card {
    flex: 1;
    min-width: 110px;
    padding: 18px 16px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid transparent;
    transition: transform 0.2s ease;
    backdrop-filter: blur(10px);
}

.summary-card:hover { transform: translateY(-3px); }

.card-icon {
    font-size: 22px;
    margin-bottom: 8px;
    display: block;
}

.card-count {
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 5px;
}

.card-type {
    font-size: 10px;
    font-weight: 700;
    color: #4a5568;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.card-am {
    font-family: 'Noto Sans Ethiopic', sans-serif;
    font-size: 13px;
    color: #4a5568;
    margin-top: 3px;
}

/* ── Legend ── */
.legend-wrapper {
    margin-top: 20px;
    padding: 20px 24px;
    background: linear-gradient(145deg, #0d1220, #111827);
    border: 1px solid #1a2235;
    border-radius: 14px;
}

.legend-row {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    justify-content: center;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: #4a5568;
    transition: color 0.2s;
}

.legend-item:hover { color: #718096; }

.legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 4px;
    flex-shrink: 0;
}

/* ── Examples ── */
.examples-header { color: #2d3748 !important; font-size: 11px !important; letter-spacing: 1px !important; }

.example-btn {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid #1a2235 !important;
    color: #4a5568 !important;
    border-radius: 8px !important;
    font-family: 'Noto Sans Ethiopic', sans-serif !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
    padding: 8px 14px !important;
}

.example-btn:hover {
    border-color: #2d3748 !important;
    color: #a0aec0 !important;
    background: rgba(255,255,255,0.06) !important;
}

.empty, .no-entity {
    color: #2d3748;
    font-size: 14px;
    text-align: center;
    padding: 24px;
    font-style: italic;
}

label { color: #2d3748 !important; font-size: 12px !important; letter-spacing: 0.5px !important; }

/* Stats bar */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 32px;
    padding: 16px 24px;
    margin-top: 20px;
    background: rgba(0,0,0,0.2);
    border-radius: 12px;
    border: 1px solid #1a2235;
}

.stat-item {
    text-align: center;
}

.stat-value {
    font-size: 18px;
    font-weight: 600;
    color: #e2e8f0;
}

.stat-label {
    font-size: 10px;
    color: #4a5568;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 2px;
}
"""

# ── Header HTML ───────────────────────────────────────────
header_html = """
<div class='app-header'>
    <span class='app-flag'>🇪🇹</span>
    <h1 class='app-title'>Amharic Named Entity Recognition</h1>
    <p class='app-subtitle'>የአማርኛ ስም ወይም ቦታ ወይም ድርጅት ለይቶ ለማወቅ — Powered by AfroXLM-R</p>
    <div class='badge-row'>
        <span class='app-badge'><b>MasakhaNER 1.0</b></span>
        <span class='app-badge blue'><b>AfroXLM-R-base</b></span>
    </div>
</div>
"""

legend_html = """
<div class='legend-wrapper'>
    <div class='legend-row'>
        <div class='legend-item'>
            <div class='legend-dot' style='background:#e53e3e;box-shadow:0 0 8px rgba(229,62,62,0.5)'></div>
            PER &mdash; Person &nbsp;(ሰው)
        </div>
        <div class='legend-item'>
            <div class='legend-dot' style='background:#38a169;box-shadow:0 0 8px rgba(56,161,105,0.5)'></div>
            LOC &mdash; Location &nbsp;(ቦታ)
        </div>
        <div class='legend-item'>
            <div class='legend-dot' style='background:#3182ce;box-shadow:0 0 8px rgba(49,130,206,0.5)'></div>
            ORG &mdash; Organization &nbsp;(ድርጅት)
        </div>
        <div class='legend-item'>
            <div class='legend-dot' style='background:#d69e2e;box-shadow:0 0 8px rgba(214,158,46,0.5)'></div>
            DATE &mdash; Date &nbsp;(ቀን)
        </div>
    </div>
</div>
"""

examples = [
    ["የአፍሪካ ኅብረት ጉባኤ በአዲስ አበባ ተካሄደ"],
    ["ነልሰን ማንዴላ በደቡብ አፍሪካ የነፃነት ታጋይ ነበሩ"],
    ["ዶ/ር ቴዎድሮስ አድሃኖም የዓለም ጤና ድርጅት ዋና ዳይሬክተር ናቸው"],
    ["የኢትዮጵያ አየር መንገድ አዲስ አበባን ከናይሮቢ ጋር ያገናኛል"],
]

# ── Build App ─────────────────────────────────────────────
with gr.Blocks(css=css, title="🇪🇹 Amharic NER") as demo:
    gr.HTML(header_html)

    with gr.Column(elem_classes="input-wrapper"):
        input_text = gr.Textbox(
            label="INPUT — የአማርኛ ጽሑፍ ያስገቡ",
            placeholder="ማንኛውንም የአማርኛ ዓረፍተ ነገር ይጻፉ...",
            lines=4,
        )
        submit_btn = gr.Button("🔍  Analyze Text — ጽሑፍ ይለኩ", variant="primary")

    with gr.Column(elem_classes="output-card"):
        output_html = gr.HTML(label="RESULTS — ውጤት")

    gr.HTML(legend_html)

    gr.Examples(
        examples=examples,
        inputs=input_text,
        label="EXAMPLE SENTENCES — ምሳሌ ዓረፍተ ነገሮች",
        elem_id="examples"
    )

    submit_btn.click(fn=predict_ner, inputs=input_text, outputs=output_html)
    input_text.submit(fn=predict_ner, inputs=input_text, outputs=output_html)

demo.launch(share=True)
