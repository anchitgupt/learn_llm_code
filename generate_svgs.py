import os

OUT_DIR = "docs/assets/visuals/triptychs"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "bg": "#13161f",
    "panel": "#181c27",
    "accent": "#5ef0c0",
    "accent_dim": "#2f8f74",
    "faint": "#5d6470",
    "ink": "#ece8de",
    "muted": "#9aa1ad",
    "amber": "#f2b65e",
    "rose": "#f08a8a",
    "line": "rgba(255,255,255,0.12)",
    "blue": "#8fd0ff"
}

def svg_header():
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 400" width="100%" height="100%">
    <defs>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&amp;display=swap');
            text {{ font-family: 'IBM Plex Mono', monospace; font-size: 16px; fill: {COLORS["ink"]}; }}
            .title {{ font-size: 24px; font-weight: 600; fill: {COLORS["accent"]}; }}
            .panel-title {{ font-size: 18px; fill: {COLORS["muted"]}; letter-spacing: 2px; text-transform: uppercase; }}
            .label {{ font-size: 14px; fill: {COLORS["muted"]}; }}
            .code {{ fill: {COLORS["amber"]}; font-size: 14px; }}
            .box {{ fill: {COLORS["panel"]}; stroke: {COLORS["line"]}; stroke-width: 2; rx: 8; }}
            .box-accent {{ fill: rgba(94, 240, 192, 0.1); stroke: {COLORS["accent"]}; stroke-width: 2; rx: 8; }}
            .arrow {{ stroke: {COLORS["faint"]}; stroke-width: 2; fill: none; }}
            .arrow-head {{ fill: {COLORS["faint"]}; }}
            .arrow-accent {{ stroke: {COLORS["accent"]}; stroke-width: 2; fill: none; }}
            .arrow-head-accent {{ fill: {COLORS["accent"]}; }}
            .node {{ fill: {COLORS["bg"]}; stroke: {COLORS["line"]}; stroke-width: 2; rx: 50; }}
            .node-active {{ fill: {COLORS["accent_dim"]}; stroke: {COLORS["accent"]}; stroke-width: 2; rx: 50; }}
        </style>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" class="arrow-head"/>
        </marker>
        <marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" class="arrow-head-accent"/>
        </marker>
    </defs>
    <rect width="100%" height="100%" fill="{COLORS["bg"]}" rx="12"/>
    <g transform="translate(40, 40)">
'''

def svg_footer():
    return '''    </g>
</svg>'''

def panel_bg(x, title):
    return f'''
    <!-- Panel -->
    <rect x="{x}" y="40" width="340" height="280" class="box" />
    <text x="{x + 20}" y="70" class="panel-title">{title}</text>
    <line x1="{x}" y1="90" x2="{x+340}" y2="90" stroke="{COLORS['line']}" stroke-width="2"/>
    '''

def draw_arrow(x1, y1, x2, y2, accent=False):
    c = "arrow-accent" if accent else "arrow"
    m = "url(#arrow-accent)" if accent else "url(#arrow)"
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{c}" marker-end="{m}"/>'

def generate_svg_00():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">00. Neural Network Foundation</text>'
    
    # Panel 1: XOR Data
    s += panel_bg(0, "1. THE PROBLEM: XOR")
    s += f'''
    <circle cx="100" cy="180" r="15" fill="{COLORS['rose']}"/>
    <text x="85" y="215" class="label">[0,0]=0</text>
    <circle cx="240" cy="180" r="15" fill="{COLORS['rose']}"/>
    <text x="225" y="215" class="label">[1,1]=0</text>
    
    <circle cx="100" cy="250" r="15" fill="{COLORS['accent']}"/>
    <text x="85" y="285" class="label">[0,1]=1</text>
    <circle cx="240" cy="250" r="15" fill="{COLORS['accent']}"/>
    <text x="225" y="285" class="label">[1,0]=1</text>
    <line x1="60" y1="120" x2="280" y2="300" stroke="{COLORS['faint']}" stroke-width="2" stroke-dasharray="5,5"/>
    <text x="30" y="320" class="label" fill="{COLORS['rose']}">Not linearly separable</text>
    '''
    
    # Panel 2: Architecture
    s += panel_bg(390, "2. THE ARCHITECTURE")
    s += f'''
    <!-- Input -->
    <rect x="420" y="140" width="40" height="30" class="node"/>
    <rect x="420" y="220" width="40" height="30" class="node"/>
    
    <!-- Hidden -->
    <rect x="530" y="120" width="40" height="30" class="node-active"/>
    <rect x="530" y="180" width="40" height="30" class="node-active"/>
    <rect x="530" y="240" width="40" height="30" class="node-active"/>
    
    <!-- Output -->
    <rect x="640" y="180" width="40" height="30" class="node-active"/>
    
    <!-- Connections -->
    {draw_arrow(460, 155, 530, 135)}
    {draw_arrow(460, 155, 530, 195)}
    {draw_arrow(460, 155, 530, 255)}
    {draw_arrow(460, 235, 530, 135)}
    {draw_arrow(460, 235, 530, 195)}
    {draw_arrow(460, 235, 530, 255)}
    
    {draw_arrow(570, 135, 640, 195)}
    {draw_arrow(570, 195, 640, 195)}
    {draw_arrow(570, 255, 640, 195)}
    
    <text x="510" y="300" class="label">Hidden Layer solves it</text>
    '''
    
    # Panel 3: Training
    s += panel_bg(780, "3. BACKPROPAGATION")
    s += f'''
    <rect x="820" y="130" width="260" height="150" class="box-accent"/>
    <text x="840" y="160" class="code">1. Forward: Pred = Sigmoid(Wx+b)</text>
    <text x="840" y="200" class="code">2. Loss = (Pred - Target)²</text>
    <text x="840" y="240" class="code">3. Backward: Chain Rule</text>
    <text x="840" y="260" class="code">   ∂L/∂W = (Pred-Target) * ...</text>
    {draw_arrow(1050, 190, 1050, 140, accent=True)}
    <text x="1060" y="170" class="label" fill="{COLORS['accent']}">Gradient Nudge</text>
    '''
    s += svg_footer()
    return s

def generate_svg_01():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">01. Bigram Counts</text>'
    s += panel_bg(0, "1. THE CONCEPT")
    s += f'''
    <text x="30" y="150" class="code">"hello"</text>
    <text x="30" y="190" class="label">Pairs:</text>
    <text x="30" y="220" class="code">h -> e</text>
    <text x="30" y="250" class="code">e -> l</text>
    <text x="30" y="280" class="code">l -> l</text>
    <text x="30" y="310" class="code">l -> o</text>
    '''
    s += panel_bg(390, "2. FREQUENCY TABLE")
    s += f'''
    <rect x="420" y="120" width="280" height="160" class="box"/>
    <line x1="420" y1="160" x2="700" y2="160" stroke="{COLORS['line']}"/>
    <text x="440" y="145" class="label">Char</text>
    <text x="540" y="145" class="label">Next Probabilities</text>
    <text x="450" y="185" class="code">h</text>
    <text x="540" y="185" class="code">e: 80%, a: 10%, i: 10%</text>
    <text x="450" y="225" class="code">e</text>
    <text x="540" y="225" class="code">l: 50%, r: 30%, ...</text>
    <text x="450" y="265" class="code">l</text>
    <text x="540" y="265" class="code">l: 60%, o: 20%, ...</text>
    '''
    s += panel_bg(780, "3. GENERATION")
    s += f'''
    <text x="820" y="140" class="label">Start with 'h'</text>
    {draw_arrow(820, 150, 820, 180)}
    <text x="820" y="200" class="label">Sample from P(next | 'h')</text>
    {draw_arrow(820, 210, 820, 240)}
    <text x="820" y="260" class="label">Got 'e' -> Sample P(next | 'e')</text>
    <text x="820" y="300" class="code" fill="{COLORS['accent']}">Output: "hello"</text>
    '''
    s += svg_footer()
    return s

def generate_svg_02():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">02. Bigram Neural Network</text>'
    s += panel_bg(0, "1. ONE-HOT INPUT")
    s += f'''
    <text x="30" y="140" class="label">Character: 'h'</text>
    <rect x="30" y="160" width="280" height="40" class="box"/>
    <text x="40" y="185" class="code">[0, 0, 0, 1, 0, ..., 0]</text>
    <text x="30" y="240" class="label">Vocabulary Size = 27</text>
    '''
    s += panel_bg(390, "2. LINEAR LAYER + SOFTMAX")
    s += f'''
    <rect x="420" y="120" width="100" height="40" class="box-accent"/>
    <text x="435" y="145" class="code">W (27x27)</text>
    {draw_arrow(520, 140, 560, 140)}
    <rect x="560" y="120" width="100" height="40" class="box"/>
    <text x="575" y="145" class="code">Logits</text>
    {draw_arrow(610, 160, 610, 200)}
    <rect x="560" y="200" width="100" height="40" class="box-accent"/>
    <text x="575" y="225" class="code">Softmax</text>
    {draw_arrow(610, 240, 610, 280)}
    <rect x="560" y="280" width="100" height="40" class="box"/>
    <text x="575" y="305" class="code">Probs</text>
    '''
    s += panel_bg(780, "3. CROSS-ENTROPY LOSS")
    s += f'''
    <text x="800" y="140" class="label">Target: 'e'</text>
    <rect x="800" y="160" width="280" height="40" class="box"/>
    <text x="810" y="185" class="code">[0, 0, 1, 0, 0, ..., 0]</text>
    <text x="800" y="240" class="code">Loss = -log(Prob['e'])</text>
    <text x="800" y="280" class="label">Gradient pulls W to match counts</text>
    '''
    s += svg_footer()
    return s

def generate_svg_03():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">03. Tokenizer (BPE)</text>'
    s += panel_bg(0, "1. CHARACTERS")
    s += f'''
    <text x="30" y="150" class="code">['t', 'h', 'e', ' ', 't', 'h', 'e', 'm', 'e']</text>
    <text x="30" y="200" class="label">Length: 9 tokens</text>
    <text x="30" y="250" class="label">Most frequent pair: ('t', 'h')</text>
    '''
    s += panel_bg(390, "2. MERGING")
    s += f'''
    <text x="420" y="140" class="label">Merge 1: ('t', 'h') -> 'th'</text>
    <text x="420" y="170" class="code">['th', 'e', ' ', 'th', 'e', 'm', 'e']</text>
    <text x="420" y="220" class="label">Merge 2: ('th', 'e') -> 'the'</text>
    <text x="420" y="250" class="code">['the', ' ', 'the', 'm', 'e']</text>
    <text x="420" y="300" class="label">Merge 3: (' ', 'the') -> ' the'</text>
    '''
    s += panel_bg(780, "3. BPE RESULT")
    s += f'''
    <text x="820" y="150" class="code">['the', ' the', 'm', 'e']</text>
    <text x="820" y="200" class="label">Length: 4 tokens</text>
    <rect x="820" y="230" width="260" height="60" class="box-accent"/>
    <text x="830" y="255" class="code">Fewer tokens = More context</text>
    <text x="830" y="275" class="code">window capacity for LLM</text>
    '''
    s += svg_footer()
    return s

def generate_svg_04():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">04. Embeddings &amp; Context (MLP LM)</text>'
    s += panel_bg(0, "1. LOOKUP EMBEDDINGS")
    s += f'''
    <text x="30" y="140" class="label">Context: ['l', 'u', 'c']</text>
    <rect x="30" y="160" width="80" height="30" class="box-accent"/>
    <rect x="120" y="160" width="80" height="30" class="box-accent"/>
    <rect x="210" y="160" width="80" height="30" class="box-accent"/>
    <text x="50" y="180" class="code">Vec(l)</text>
    <text x="140" y="180" class="code">Vec(u)</text>
    <text x="230" y="180" class="code">Vec(c)</text>
    <text x="30" y="230" class="label">Each char mapped to dense vector</text>
    '''
    s += panel_bg(390, "2. FLATTEN &amp; MLP")
    s += f'''
    <text x="420" y="140" class="label">Concat</text>
    <rect x="420" y="160" width="280" height="30" class="box"/>
    {draw_arrow(560, 190, 560, 220)}
    <rect x="420" y="220" width="280" height="30" class="box-accent"/>
    <text x="490" y="240" class="code">Hidden (Tanh)</text>
    {draw_arrow(560, 250, 560, 280)}
    <rect x="420" y="280" width="280" height="30" class="box"/>
    <text x="500" y="300" class="code">Logits</text>
    '''
    s += panel_bg(780, "3. PREDICTION")
    s += f'''
    <text x="820" y="150" class="code">Context Window = 3</text>
    <text x="820" y="190" class="label">Prediction sees 3 chars jointly</text>
    <text x="820" y="230" class="label">Generates real-looking names:</text>
    <text x="820" y="270" class="code" fill="{COLORS['accent']}">"lucas", "isabella"</text>
    '''
    s += svg_footer()
    return s

def generate_svg_05():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">05. Autograd (Automatic Differentiation)</text>'
    s += panel_bg(0, "1. COMPUTE GRAPH")
    s += f'''
    <circle cx="100" cy="180" r="20" class="node"/>
    <text x="95" y="185" class="code">w</text>
    <circle cx="100" cy="240" r="20" class="node"/>
    <text x="95" y="245" class="code">x</text>
    <circle cx="200" cy="210" r="20" class="node-active"/>
    <text x="195" y="215" class="code">*</text>
    <circle cx="280" cy="210" r="20" class="node"/>
    <text x="275" y="215" class="code">y</text>
    {draw_arrow(120, 180, 180, 205)}
    {draw_arrow(120, 240, 180, 215)}
    {draw_arrow(220, 210, 260, 210)}
    '''
    s += panel_bg(390, "2. FORWARD PASS")
    s += f'''
    <text x="420" y="150" class="code">w.data = 2.0</text>
    <text x="420" y="180" class="code">x.data = -3.0</text>
    <text x="420" y="230" class="label">Computes y = w * x</text>
    <text x="420" y="260" class="code">y.data = -6.0</text>
    <text x="420" y="300" class="label">Saves pointers to parents</text>
    '''
    s += panel_bg(780, "3. BACKWARD PASS")
    s += f'''
    <text x="800" y="140" class="code">y.backward()</text>
    <text x="800" y="180" class="label">Walks graph backwards:</text>
    <text x="800" y="220" class="code">w.grad += y.grad * x.data</text>
    <text x="800" y="250" class="code">x.grad += y.grad * w.data</text>
    <text x="800" y="290" class="label" fill="{COLORS['accent']}">Automates the chain rule!</text>
    '''
    s += svg_footer()
    return s

def generate_svg_06():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">06. Switching to PyTorch</text>'
    s += panel_bg(0, "1. TENSORS")
    s += f'''
    <rect x="30" y="140" width="280" height="40" class="box"/>
    <text x="40" y="165" class="code">torch.randn(3, 4, requires_grad=True)</text>
    <text x="30" y="220" class="label">GPU Accelerated</text>
    <text x="30" y="260" class="label">Batched Matrix Math</text>
    '''
    s += panel_bg(390, "2. NN.MODULE")
    s += f'''
    <rect x="420" y="140" width="280" height="40" class="box-accent"/>
    <text x="430" y="165" class="code">nn.Linear(10, 20)</text>
    <text x="420" y="220" class="label">Manages parameters automatically</text>
    <text x="420" y="260" class="code">nn.Embedding</text>
    <text x="420" y="290" class="code">F.cross_entropy</text>
    '''
    s += panel_bg(780, "3. THE MAGIC LOOP")
    s += f'''
    <text x="800" y="140" class="code" fill="{COLORS['accent']}">optimizer.zero_grad()</text>
    <text x="800" y="180" class="code">logits = model(x)</text>
    <text x="800" y="220" class="code">loss = F.cross_entropy(logits, y)</text>
    <text x="800" y="260" class="code" fill="{COLORS['accent']}">loss.backward()</text>
    <text x="800" y="300" class="code" fill="{COLORS['accent']}">optimizer.step()</text>
    '''
    s += svg_footer()
    return s

def generate_svg_07():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">07. Self-Attention</text>'
    s += panel_bg(0, "1. QUERY, KEY, VALUE")
    s += f'''
    <text x="30" y="140" class="label">For each token:</text>
    <text x="30" y="180" class="code" fill="{COLORS['amber']}">Query (What I look for)</text>
    <text x="30" y="220" class="code" fill="{COLORS['blue']}">Key   (What I offer)</text>
    <text x="30" y="260" class="code" fill="{COLORS['accent']}">Value (What I pass on)</text>
    '''
    s += panel_bg(390, "2. DOT PRODUCT &amp; MASK")
    s += f'''
    <rect x="420" y="120" width="140" height="140" class="box"/>
    <text x="430" y="140" class="code">Scores (Q @ K^T)</text>
    <path d="M 420 120 L 560 260 L 560 120 Z" fill="rgba(0,0,0,0.5)" stroke="{COLORS['faint']}"/>
    <text x="480" y="160" class="label" fill="{COLORS['faint']}">-inf</text>
    <text x="440" y="220" class="label">Past</text>
    <text x="420" y="290" class="label">Causal Mask forbids peeking ahead</text>
    '''
    s += panel_bg(780, "3. WEIGHTED SUM")
    s += f'''
    <rect x="800" y="130" width="260" height="40" class="box-accent"/>
    <text x="810" y="155" class="code">Weights = Softmax(Scores)</text>
    {draw_arrow(930, 170, 930, 200)}
    <rect x="800" y="200" width="260" height="40" class="box"/>
    <text x="810" y="225" class="code">Output = Weights @ Value</text>
    <text x="800" y="280" class="label">A content-based weighted average</text>
    '''
    s += svg_footer()
    return s

def generate_svg_08():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">08. Transformer Block</text>'
    s += panel_bg(0, "1. MULTI-HEAD ATTENTION")
    s += f'''
    <rect x="30" y="140" width="60" height="60" class="box-accent"/>
    <rect x="100" y="140" width="60" height="60" class="box-accent"/>
    <rect x="170" y="140" width="60" height="60" class="box-accent"/>
    <rect x="240" y="140" width="60" height="60" class="box-accent"/>
    <text x="40" y="230" class="label">Run several attention</text>
    <text x="40" y="250" class="label">heads in parallel,</text>
    <text x="40" y="270" class="label">then concatenate.</text>
    '''
    s += panel_bg(390, "2. THE BLOCK")
    s += f'''
    <rect x="460" y="120" width="200" height="30" class="box"/>
    <text x="480" y="140" class="code">LayerNorm</text>
    {draw_arrow(560, 150, 560, 180)}
    <rect x="460" y="180" width="200" height="30" class="box-accent"/>
    <text x="480" y="200" class="code">Multi-Head Attention</text>
    {draw_arrow(560, 210, 560, 240)}
    <rect x="460" y="240" width="200" height="30" class="box"/>
    <text x="480" y="260" class="code">LayerNorm -> FFN</text>
    '''
    s += panel_bg(780, "3. RESIDUAL CONNECTIONS")
    s += f'''
    <path d="M 430 110 L 430 220 L 460 220" class="arrow-accent"/>
    <text x="800" y="150" class="code">x = x + MHA(LN(x))</text>
    <text x="800" y="190" class="code">x = x + FFN(LN(x))</text>
    <text x="800" y="240" class="label">Identity shortcuts let</text>
    <text x="800" y="260" class="label">gradients flow, making</text>
    <text x="800" y="280" class="label">deep stacks trainable.</text>
    '''
    s += svg_footer()
    return s

def generate_svg_09():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">09. Tiny GPT</text>'
    s += panel_bg(0, "1. EMBEDDINGS")
    s += f'''
    <text x="30" y="160" class="code">Token Emb + Position Emb</text>
    <rect x="30" y="180" width="280" height="40" class="box"/>
    <text x="40" y="205" class="label">Position is explicitly added</text>
    <text x="30" y="260" class="label">because attention is set-based.</text>
    '''
    s += panel_bg(390, "2. THE STACK")
    s += f'''
    <rect x="460" y="130" width="200" height="30" class="box-accent"/>
    <text x="510" y="150" class="code">Block N</text>
    <rect x="460" y="170" width="200" height="30" class="box-accent"/>
    <text x="510" y="190" class="code">Block 2</text>
    <rect x="460" y="210" width="200" height="30" class="box-accent"/>
    <text x="510" y="230" class="code">Block 1</text>
    {draw_arrow(560, 250, 560, 280)}
    <text x="470" y="300" class="code">Final LayerNorm</text>
    '''
    s += panel_bg(780, "3. LANGUAGE MODELING")
    s += f'''
    <rect x="800" y="140" width="260" height="40" class="box"/>
    <text x="810" y="165" class="code">Linear Head (Vocab Size)</text>
    <text x="800" y="220" class="label">Trained on Shakespeare</text>
    <text x="800" y="250" class="code" fill="{COLORS['accent']}">Produces authentic text format!</text>
    '''
    s += svg_footer()
    return s

def generate_svg_10():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">10. Sampling Strategies</text>'
    s += panel_bg(0, "1. TEMPERATURE")
    s += f'''
    <text x="30" y="140" class="code">Logits / Temperature</text>
    <text x="30" y="180" class="label">T = 0 : Greedy (stuck)</text>
    <text x="30" y="220" class="label">T &lt; 1 : Sharpen (safe)</text>
    <text x="30" y="260" class="label">T > 1 : Flatten (wild)</text>
    '''
    s += panel_bg(390, "2. TOP-K &amp; TOP-P")
    s += f'''
    <rect x="420" y="130" width="280" height="80" class="box"/>
    <text x="430" y="160" class="code">Top-K: keep k highest</text>
    <text x="430" y="190" class="code">Top-P: cumulative mass >= p</text>
    <text x="420" y="250" class="label">Truncates the unreliable tail</text>
    <text x="420" y="280" class="label">of the probability distribution.</text>
    '''
    s += panel_bg(780, "3. SHAPING GENERATION")
    s += f'''
    <text x="800" y="150" class="label">Model weights NEVER change.</text>
    <text x="800" y="190" class="label">We only shape how we sample</text>
    <text x="800" y="220" class="label">from the output distribution.</text>
    <rect x="800" y="250" width="260" height="40" class="box-accent"/>
    <text x="810" y="275" class="code">API knobs: temp, top_p</text>
    '''
    s += svg_footer()
    return s

def generate_svg_11():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">11. Production Training</text>'
    s += panel_bg(0, "1. LEARNING RATE SCHEDULE")
    s += f'''
    <path d="M 30 260 L 80 140 L 280 260" fill="none" stroke="{COLORS['accent']}" stroke-width="3"/>
    <line x1="30" y1="260" x2="300" y2="260" stroke="{COLORS['line']}"/>
    <line x1="30" y1="260" x2="30" y2="120" stroke="{COLORS['line']}"/>
    <text x="40" y="130" class="label">Warmup</text>
    <text x="200" y="200" class="label">Cosine Decay</text>
    '''
    s += panel_bg(390, "2. CHECKPOINTING")
    s += f'''
    <rect x="420" y="140" width="280" height="120" class="box"/>
    <text x="440" y="170" class="code">Save state dicts:</text>
    <text x="440" y="200" class="code">- Model weights</text>
    <text x="440" y="230" class="code">- Optimizer momentum</text>
    <text x="420" y="290" class="label" fill="{COLORS['accent']}">Resumable across crashes!</text>
    '''
    s += panel_bg(780, "3. ROBUSTNESS")
    s += f'''
    <text x="800" y="150" class="code">Gradient Clipping:</text>
    <text x="800" y="180" class="label">Caps large gradients.</text>
    <text x="800" y="230" class="code">AdamW:</text>
    <text x="800" y="260" class="label">Decoupled weight decay.</text>
    '''
    s += svg_footer()
    return s

def generate_svg_12():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">12. Instruction Tuning (SFT)</text>'
    s += panel_bg(0, "1. BASE MODEL")
    s += f'''
    <text x="30" y="140" class="label">Learns P(text)</text>
    <text x="30" y="180" class="code">Prompt: "say hello"</text>
    <text x="30" y="220" class="label">Result (Continues text):</text>
    <text x="30" y="260" class="code" fill="{COLORS['rose']}">"say hello to the world and..."</text>
    '''
    s += panel_bg(390, "2. SFT DATA FORMAT")
    s += f'''
    <rect x="420" y="130" width="280" height="120" class="box-accent"/>
    <text x="430" y="160" class="code">User: say hello</text>
    <text x="430" y="190" class="code">Asst: good morrow to you.</text>
    <text x="420" y="280" class="label">Trained on Q&amp;A pairs.</text>
    '''
    s += panel_bg(780, "3. LOSS MASKING")
    s += f'''
    <text x="800" y="140" class="label">Mask out User prompt:</text>
    <text x="800" y="180" class="code">User: say hello (Loss = 0)</text>
    <text x="800" y="220" class="label">Train ONLY on answer:</text>
    <text x="800" y="260" class="code">Asst: good morrow (Loss ON)</text>
    <text x="800" y="300" class="label" fill="{COLORS['accent']}">Model learns to ANSWER.</text>
    '''
    s += svg_footer()
    return s

def generate_svg_13():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">13. DPO (Direct Preference Optimization)</text>'
    s += panel_bg(0, "1. PREFERENCE DATA")
    s += f'''
    <text x="30" y="140" class="code">Prompt: "Help me code"</text>
    <rect x="30" y="160" width="280" height="50" class="box-accent"/>
    <text x="40" y="190" class="code" fill="{COLORS['accent']}">Chosen (Good answer)</text>
    <rect x="30" y="220" width="280" height="50" class="box"/>
    <text x="40" y="250" class="code" fill="{COLORS['rose']}">Rejected (Bad answer)</text>
    '''
    s += panel_bg(390, "2. NO REWARD MODEL")
    s += f'''
    <text x="420" y="140" class="label">RLHF needs 3 models + PPO</text>
    <line x1="420" y1="160" x2="680" y2="160" stroke="{COLORS['rose']}" stroke-width="2"/>
    <text x="420" y="200" class="code" fill="{COLORS['accent']}">DPO uses 1 Loss Function</text>
    <text x="420" y="240" class="label">Optimize preferences directly!</text>
    <text x="420" y="280" class="label">Simple binary classification.</text>
    '''
    s += panel_bg(780, "3. THE MECHANISM")
    s += f'''
    <text x="800" y="140" class="label">Policy Model vs Frozen Ref</text>
    <text x="800" y="180" class="code">Maximize: log(Policy(Chosen))</text>
    <text x="800" y="220" class="code">Minimize: log(Policy(Rejected))</text>
    <text x="800" y="280" class="label">Implicit KL constraint keeps</text>
    <text x="800" y="300" class="label">model from drifting too far.</text>
    '''
    s += svg_footer()
    return s

def generate_svg_14():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">14. Modern GPT Architecture (2026)</text>'
    s += panel_bg(0, "1. RoPE &amp; RMSNorm")
    s += f'''
    <text x="30" y="140" class="code">RoPE (Rotary Embeddings)</text>
    <text x="30" y="170" class="label">Rotate Q &amp; K vectors.</text>
    <text x="30" y="200" class="label">No learned position tables!</text>
    <text x="30" y="250" class="code">RMSNorm</text>
    <text x="30" y="280" class="label">Normalize by Root Mean Square.</text>
    <text x="30" y="310" class="label">Drops mean/bias = Cheaper.</text>
    '''
    s += panel_bg(390, "2. SwiGLU")
    s += f'''
    <rect x="420" y="130" width="280" height="80" class="box-accent"/>
    <text x="430" y="160" class="code">Gated Feed-Forward:</text>
    <text x="430" y="190" class="code">SiLU(xW1) * xW3</text>
    <text x="420" y="250" class="label">More expressive power per</text>
    <text x="420" y="280" class="label">parameter than standard ReLU.</text>
    '''
    s += panel_bg(780, "3. GQA")
    s += f'''
    <text x="800" y="140" class="code">Grouped-Query Attention</text>
    <text x="800" y="180" class="label">Multiple Query heads share</text>
    <text x="800" y="210" class="label">a single Key/Value head.</text>
    <rect x="800" y="240" width="260" height="50" class="box-accent"/>
    <text x="810" y="270" class="code">Lower Loss, Fewer Params!</text>
    '''
    s += svg_footer()
    return s

def generate_svg_15():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">15. The KV Cache</text>'
    s += panel_bg(0, "1. NAIVE GENERATION")
    s += f'''
    <text x="30" y="150" class="label">Recomputes entire context.</text>
    <text x="30" y="190" class="code" fill="{COLORS['rose']}">O(T²) Work</text>
    <text x="30" y="230" class="label">Step 1: [A] -> K, V</text>
    <text x="30" y="260" class="label">Step 2: [A, B] -> K, V</text>
    <text x="30" y="290" class="label">Step 3: [A, B, C] -> K, V</text>
    '''
    s += panel_bg(390, "2. CACHING")
    s += f'''
    <text x="420" y="150" class="label">Past keys/values NEVER change.</text>
    <text x="420" y="190" class="code" fill="{COLORS['accent']}">O(T) Work</text>
    <rect x="420" y="220" width="280" height="60" class="box-accent"/>
    <text x="430" y="255" class="code">Cache: [K_A, K_B]</text>
    <text x="420" y="310" class="label">Just append K_C and attend!</text>
    '''
    s += panel_bg(780, "3. GQA PAYOFF")
    s += f'''
    <text x="800" y="140" class="label">Memory = L * n_kv * d * T</text>
    <text x="800" y="190" class="label">GQA has fewer n_kv heads.</text>
    <rect x="800" y="220" width="260" height="70" class="box-accent"/>
    <text x="810" y="250" class="code">GQA: 48 KB</text>
    <text x="810" y="275" class="code">MHA: 96 KB</text>
    '''
    s += svg_footer()
    return s

def generate_svg_16():
    s = svg_header()
    s += f'<text x="0" y="10" class="title">16. Compressing the Cache (TurboQuant)</text>'
    s += panel_bg(0, "1. THE PROBLEM")
    s += f'''
    <text x="30" y="140" class="label">Long context = huge cache.</text>
    <text x="30" y="180" class="label">Naive quantization fails</text>
    <text x="30" y="210" class="label">because of outlier channels.</text>
    <line x1="30" y1="240" x2="280" y2="240" stroke="{COLORS['line']}"/>
    <circle cx="200" cy="240" r="10" fill="{COLORS['rose']}"/>
    <text x="30" y="280" class="code" fill="{COLORS['rose']}">Outlier destroys scale grid</text>
    '''
    s += panel_bg(390, "2. THE TRICK")
    s += f'''
    <text x="420" y="140" class="code">Rotate by Orthogonal Matrix</text>
    <text x="420" y="180" class="label">Rotations preserve dot products</text>
    <text x="420" y="210" class="label">(so attention scores stay same).</text>
    <text x="420" y="250" class="label">Spreads outlier energy across</text>
    <text x="420" y="280" class="label">all coordinates evenly.</text>
    '''
    s += panel_bg(780, "3. 3-BIT KV CACHE")
    s += f'''
    <rect x="800" y="130" width="260" height="50" class="box"/>
    <text x="810" y="160" class="code">Rotate -> Quantize -> 3-bit</text>
    <text x="800" y="220" class="label">Near-zero loss in fidelity.</text>
    <rect x="800" y="240" width="260" height="50" class="box-accent"/>
    <text x="810" y="270" class="code">5.3x Compression!</text>
    '''
    s += svg_footer()
    return s

generators = [
    ("00", generate_svg_00),
    ("01", generate_svg_01),
    ("02", generate_svg_02),
    ("03", generate_svg_03),
    ("04", generate_svg_04),
    ("05", generate_svg_05),
    ("06", generate_svg_06),
    ("07", generate_svg_07),
    ("08", generate_svg_08),
    ("09", generate_svg_09),
    ("10", generate_svg_10),
    ("11", generate_svg_11),
    ("12", generate_svg_12),
    ("13", generate_svg_13),
    ("14", generate_svg_14),
    ("15", generate_svg_15),
    ("16", generate_svg_16),
]

for idx, gen in generators:
    with open(f"{OUT_DIR}/{idx}.svg", "w") as f:
        f.write(gen())
print("SVGs generated.")
