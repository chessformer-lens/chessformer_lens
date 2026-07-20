"""
ui.py — the entire interface (HTML + CSS + JS) as one string.

Pure markup: no model, and the only Python is the piece-set injection at the
bottom (pieces.py -> `PIECE_URI`). The page talks to `MaiaApi` (bridge.py)
over `window.pywebview.api`. Edit the app's look and front-end behavior here.
"""

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chessformer (Maia 3) Interpretability</title>
<style>
  :root{
    --bg:#0e1014; --panel:#161a21; --panel2:#1b2029; --line:#262c37;
    --text:#e7eaf0; --muted:#8b93a3; --accent:#6ea8fe; --accent2:#7bd88f;
    --sq-light:#c9d1dc; --sq-dark:#6b7686;
    --hl:rgba(245,213,107,.28); --sel:#7bd88f; --win:#5fb878; --draw:#6b7480; --loss:#d9606a;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;}
  .wrap{display:flex;gap:18px;padding:16px 20px;height:100%;align-items:flex-start;overflow:hidden}
  .left{display:flex;flex-direction:column;gap:10px}
  .right{flex:1;display:flex;flex-direction:column;gap:12px;min-width:260px;max-width:340px;height:100%}
  .arch{flex:0 0 384px;display:flex;flex-direction:column;height:100%}

  h1{font-size:15px;font-weight:600;letter-spacing:.3px;margin:0}
  .sub{font-size:11px;color:var(--muted);font-family:var(--mono);margin-top:3px}

  /* board — capped at 640px but shrinks to keep the header, the board and the
     toggle button under it all inside the viewport (the page never scrolls) */
  #boardwrap{position:relative;width:min(640px, calc(100vh - 148px));
    height:min(640px, calc(100vh - 148px))}
  #board{width:100%;height:100%;display:grid;grid-template-columns:repeat(8,1fr);
    grid-template-rows:repeat(8,1fr);border-radius:8px;overflow:hidden;
    box-shadow:0 10px 40px rgba(0,0,0,.45);user-select:none}
  #arrowsvg{position:absolute;inset:0;width:100%;height:100%;z-index:6;pointer-events:none}
  .sq{position:relative;display:flex;align-items:center;justify-content:center;cursor:default}
  .sq.light{background:var(--sq-light)} .sq.dark{background:var(--sq-dark)}
  .sq.lastmove::after{content:"";position:absolute;inset:0;background:var(--hl)}
  .sq.sel{box-shadow:inset 0 0 0 4px var(--sel)}
  .sq img.pc{position:relative;z-index:2;width:87%;height:87%;
    filter:drop-shadow(0 2px 2px rgba(0,0,0,.30));pointer-events:none}
  .sq .dot{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
    width:30%;height:30%;border-radius:50%;background:rgba(40,50,40,.42);z-index:1;pointer-events:none}
  .sq.cap .dot{width:86%;height:86%;background:transparent;
    box-shadow:inset 0 0 0 4px rgba(40,50,40,.40)}
  .sq.playable{cursor:pointer}
  .sq.attq::before{content:"";position:absolute;inset:2px;border:2px solid rgba(110,168,254,.75);border-radius:4px;z-index:1;pointer-events:none}
  .coord{position:absolute;font-size:9px;font-family:var(--mono);color:rgba(20,24,30,.55);z-index:3}
  .coord.f{right:3px;bottom:2px} .coord.r{left:3px;top:2px}

  .controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  button,select{background:var(--panel2);color:var(--text);border:1px solid var(--line);
    border-radius:7px;padding:7px 11px;font-size:12px;cursor:pointer}
  button:hover{border-color:var(--accent)}
  button.primary{background:var(--accent);color:#0a1220;border-color:var(--accent);font-weight:600}
  label.lbl{font-size:11px;color:var(--muted)}

  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
  .card h2{margin:0 0 10px;font-size:12px;font-weight:600;color:var(--muted);
    text-transform:uppercase;letter-spacing:.6px}

  /* elo slider */
  .elorow{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:8px}
  .eloval{font-family:var(--mono);font-size:26px;font-weight:600;color:var(--accent)}
  input[type=range]{-webkit-appearance:none;width:100%;height:5px;border-radius:4px;
    background:linear-gradient(90deg,var(--accent2),var(--accent));outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;
    border-radius:50%;background:#fff;border:3px solid var(--accent);cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,.4)}
  .ticks{display:flex;justify-content:space-between;font-size:9px;color:var(--muted);
    font-family:var(--mono);margin-top:5px}
  .cmptoggle{display:block;margin-top:10px;cursor:pointer}
  #cmpbox{margin-top:8px}
  #cmpbox.hidden{display:none}
  .eloval2{font-family:var(--mono);font-size:16px;font-weight:600;color:var(--accent2)}

  /* wdl */
  .wdl{display:flex;height:22px;border-radius:6px;overflow:hidden;font-size:10px;
    font-family:var(--mono);color:#0d130d}
  .wdl div{display:flex;align-items:center;justify-content:center;min-width:0}
  .wdl .w{background:var(--win)} .wdl .d{background:#6b7480;color:#10141a} .wdl .l{background:var(--loss)}
  .wdl.cmp{flex-direction:column;height:42px;gap:2px;font-size:9px}
  .wdl .wdlrow{display:flex;flex:1;min-width:0;justify-content:flex-start;border-radius:4px;overflow:hidden}
  .wdltag{flex:0 0 30px;display:flex;align-items:center;padding-left:2px;
    color:var(--muted);font-family:var(--mono);background:#0f131a}

  /* policy list */
  #policy{flex:1;overflow-y:auto;min-height:0}
  .prow{display:grid;grid-template-columns:52px 1fr 52px;align-items:center;gap:10px;
    padding:4px 0;font-size:12px}
  .prow .san{font-family:var(--mono);color:var(--text)}
  .prow .barwrap{display:block;height:18px;background:#0f131a;border:1px solid var(--line);
    border-radius:10px;overflow:hidden}
  .prow .bar{display:block;height:100%;min-width:3px;border-radius:10px;
    background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .18s ease}
  .prow .pct{font-family:var(--mono);text-align:right;color:var(--muted)}
  .prow.top .san{color:var(--accent2);font-weight:700}
  .prow.top .pct{color:var(--accent2)}
  .prow.played .barwrap{box-shadow:0 0 0 2px var(--hl)}
  /* compare mode: one thin bar per rating + signed delta */
  .prow.cmp .pct{font-size:11px}
  .dualbar{display:flex;flex-direction:column;gap:2px;justify-content:center;min-width:0}
  .dualbar .bar{display:block;height:7px;min-width:2px;border-radius:4px}
  .dualbar .bar.a{background:var(--accent)}
  .dualbar .bar.b{background:var(--accent2)}
  .delta.up{color:var(--accent2)} .delta.down{color:var(--loss)}

  /* architecture diagram */
  .diagram{padding:10px 12px}
  .diagram svg{display:block;width:100%;height:auto}

  /* live attention panel */
  .attctrls{display:flex;flex-direction:column;gap:8px;margin-bottom:10px}
  .chiprow{display:flex;align-items:flex-start;gap:8px}
  .chiprow .lbl{flex:0 0 38px;padding-top:4px}
  .chips{display:flex;flex:1;min-width:0;flex-wrap:wrap;gap:4px}
  .chip{padding:3px 8px;font-size:11px;border:1px solid var(--line);border-radius:6px;
    background:var(--panel2);cursor:pointer;font-family:var(--mono);color:var(--text)}
  .chip:hover{border-color:var(--accent)}
  .chip.active{background:var(--accent);color:#0a1220;border-color:var(--accent);font-weight:700}
  /* single-head ablation */
  .ablrow{display:flex;align-items:center;gap:8px;margin-top:2px}
  #ablbtn{padding:4px 10px;font-size:11px}
  .ablnote{font-size:9px;color:var(--muted);font-family:var(--mono)}
  #ablout{margin:0 0 8px;font-family:var(--mono)}
  .ablhead{font-size:9px;color:var(--muted);margin-bottom:5px;line-height:1.4}
  .ablitem{display:grid;grid-template-columns:56px 1fr 44px;gap:8px;font-size:11px;padding:2px 0}
  .ablitem .vals{color:var(--muted);text-align:right}
  .ablitem .abldelta{text-align:right}
  .abldelta.up{color:var(--accent2)} .abldelta.down{color:var(--loss)}
  .ablwdl{font-size:9px;color:var(--muted);margin-top:5px}
  .attcap{font-size:10px;color:var(--muted);margin-bottom:10px;line-height:1.45}
  .attset{display:flex;flex-direction:column;gap:12px}
  .attlabel{font-size:10px;color:var(--muted);font-family:var(--mono);margin-bottom:4px}
  .attboard{width:100%;max-width:202px;aspect-ratio:1/1;margin:0 auto;display:grid;
    grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr);
    border:1px solid var(--line);border-radius:5px;overflow:hidden;background:#10141b}
  .attcell{cursor:pointer;position:relative}
  .attcell.q{box-shadow:inset 0 0 0 2px #ff5d6c}
  /* faint checkerboard over the heat so squares stay identifiable */
  .attcell::after{content:"";position:absolute;inset:0;pointer-events:none}
  .attcell.dk::after{background:rgba(0,0,0,.16)}
  .attcell.lt::after{background:rgba(255,255,255,.05)}
  .attcoord{position:absolute;z-index:1;font-size:6px;line-height:1;font-family:var(--mono);
    color:rgba(255,255,255,.9);text-shadow:0 0 2px rgba(0,0,0,.95);pointer-events:none}
  .attcoord.f{right:1px;bottom:0} .attcoord.r{left:1px;top:0}
  .fenrow{display:flex;gap:6px;margin-top:8px}
  .fenbox{flex:1;min-width:0;background:#0f131a;border:1px solid var(--line);border-radius:6px;
    color:var(--text);font-family:var(--mono);font-size:10px;padding:5px 7px}
  #fenload{padding:5px 10px;font-size:11px}
  /* side-by-side board pairs (content vs geometry) */
  .attpair{display:flex;gap:10px}
  .attpair>div{flex:1;min-width:0}
  .attpair .attboard{max-width:none}
  /* smolgen mixture: readout + coefficient strip + template gallery */
  .smoltitle{margin:16px 0 8px;font-size:12px;font-weight:600;color:var(--muted);
    text-transform:uppercase;letter-spacing:.6px}
  .gabreadout{font-family:var(--mono);font-size:10px;line-height:1.7;background:#0f131a;
    border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin:6px 0 10px;min-height:30px}
  .gabreadout b{color:var(--text)}
  .gabreadout .pos{color:#f0a35e} .gabreadout .neg{color:#6fb3ff}
  .gabreadout .tref{color:var(--accent);cursor:pointer}
  .gabreadout .tref:hover{text-decoration:underline}
  .coeffstrip{display:flex;align-items:stretch;gap:1px;height:44px;border:1px solid var(--line);
    border-radius:6px;padding:2px;margin-bottom:12px;
    background:linear-gradient(#0f131a 49%,#262c37 49%,#262c37 51%,#0f131a 51%)}
  .cbar{flex:1;min-width:1px;position:relative;cursor:pointer}
  .cbar span{position:absolute;left:0;right:0;border-radius:1px}
  .cbar:hover{background:rgba(110,168,254,.15)}
  .cbar.selt{background:rgba(110,168,254,.3)}
  .gallery{display:grid;grid-template-columns:repeat(8,1fr);gap:4px}
  .gtile{cursor:pointer;text-align:center;min-width:0}
  .gtile canvas{display:block;width:100%;aspect-ratio:1/1;image-rendering:pixelated;
    border:1px solid var(--line);border-radius:3px;background:#10141b}
  .gtile:hover canvas{border-color:var(--accent)}
  .gtile.selt canvas{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
  .gtlbl{font-size:7px;font-family:var(--mono);color:var(--muted);margin-top:1px;
    white-space:nowrap;overflow:hidden}
  .gtlbl b{font-weight:600}
  .gabdetail{border:1px solid var(--line);border-radius:8px;background:var(--panel2);
    padding:10px;margin-bottom:12px}
  .gabdetail.hidden{display:none}
  .gdrow{display:flex;gap:12px;align-items:flex-start;justify-content:center}
  .gdrow canvas{display:block;width:140px;height:140px;image-rendering:pixelated;
    border:1px solid var(--line);border-radius:4px;background:#10141b}
  .gdboard{width:140px;flex:0 0 140px}
  .gdboard .attboard{max-width:140px}
  .gdinfo{font-family:var(--mono);font-size:10px;line-height:1.7;color:var(--muted);margin-top:8px}
  .gdinfo b{color:var(--text)}
  .gdclose{float:right;padding:2px 8px;font-size:10px}
  .attlegend{display:flex;align-items:center;gap:6px;margin-top:10px;font-size:9px;color:var(--muted);font-family:var(--mono)}
  .legbar{flex:1;height:8px;border-radius:4px;border:1px solid var(--line);
    background:linear-gradient(90deg, rgb(68,1,84), rgb(59,82,139), rgb(33,144,141), rgb(93,200,99), rgb(253,231,37))}
  .legbar.div{background:linear-gradient(90deg, rgb(64,132,234), rgb(24,28,36), rgb(244,134,58))}
  .legbar.pos{background:linear-gradient(90deg, rgb(24,28,36), rgb(244,134,58))}
  .leghint{font-size:8px;color:var(--muted);margin-top:3px;font-family:var(--mono)}

  /* residual-stream filmstrip (lives in the bottom drawer) — sized so the
     board and pieces are actually legible, not a strip of colored dots */
  .film{display:flex;gap:12px;overflow-x:auto;padding-bottom:6px}
  .filmcol{flex:0 0 150px;display:flex;flex-direction:column;align-items:center;gap:6px;min-width:0}
  .miniboard{width:100%;aspect-ratio:1/1;display:grid;grid-template-columns:repeat(8,1fr);
    grid-template-rows:repeat(8,1fr);border:1px solid var(--line);border-radius:6px;overflow:hidden;background:#10141b}
  .miniboard>div{position:relative;display:flex;align-items:center;justify-content:center;line-height:1}
  /* faint checkerboard under the heat / move markers */
  .miniboard>div.dk::after,.miniboard>div.lt::after{content:"";position:absolute;inset:0;pointer-events:none}
  .miniboard>div.dk::after{background:rgba(0,0,0,.16)}
  .miniboard>div.lt::after{background:rgba(255,255,255,.05)}
  /* moving piece drawn on the from-square of the logit-lens move */
  .miniboard img.pc{width:88%;height:88%;position:relative;z-index:3;pointer-events:none}
  .filmlbl{font-size:11px;color:var(--muted);font-family:var(--mono);text-align:center}
  /* structure tags: which module wrote this column of the stream */
  .filmcol.emb  .miniboard{border-top:3px solid #8a93a3}
  .filmcol.attn .miniboard{border-top:3px solid #f0a35e}   /* attention add */
  .filmcol.mlp  .miniboard{border-top:3px solid #6fb3ff}   /* MLP add */
  .filmcol.enc  .miniboard{border-top:3px solid #5ac878}   /* final norm = real output */
  .filmcol.emb  .filmlbl{color:#8a93a3}
  .filmcol.attn .filmlbl{color:#f0a35e}
  .filmcol.mlp  .filmlbl{color:#6fb3ff}
  .filmcol.enc  .filmlbl{color:#5ac878}
  .residlegend{display:flex;gap:12px;font-size:9px;font-family:var(--mono);margin-bottom:8px;color:var(--muted)}
  .residlegend span::before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:middle}
  .residlegend .lg-attn::before{background:#f0a35e}
  .residlegend .lg-mlp::before{background:#6fb3ff}
  .residlegend .lg-emb::before{background:#8a93a3}
  .residlegend .lg-enc::before{background:#5ac878}

  /* medium toggle button under the board */
  .boardctrls{margin-top:12px;display:flex;justify-content:center}
  .medbtn{padding:10px 22px;font-size:13px;font-weight:600;border-radius:9px}
  .medbtn.active{background:var(--accent);color:#0a1220;border-color:var(--accent)}

  /* slide-up drawers from the bottom (move microscope + residual film) */
  .drawer{position:fixed;left:0;right:0;bottom:0;z-index:40;background:var(--panel);
    border-top:1px solid var(--line);box-shadow:0 -12px 44px rgba(0,0,0,.55);
    transform:translateY(105%);transition:transform .25s ease;padding:10px 20px 14px}
  .drawer.open{transform:translateY(0)}
  .mlhead{display:flex;align-items:center;gap:12px;margin-bottom:8px}
  .mltitle{font-size:13px;font-weight:600}
  .mltitle b{color:var(--accent2)}
  .mlhint{font-size:10px;color:var(--muted);font-family:var(--mono);flex:1;min-width:0}
  #mlclose{padding:3px 10px;font-size:11px}
  .mlbody{display:flex;gap:24px;align-items:flex-start}
  .mlchart{flex:1;min-width:0}
  #mlsvg{width:100%;height:auto;display:block;background:#0f131a;
    border:1px solid var(--line);border-radius:8px}
  .mlgridbox{flex:0 0 250px}
  #ablgrid{display:grid;grid-template-columns:18px repeat(8,1fr);grid-template-rows:14px repeat(8,1fr);
    gap:1px;width:250px;margin-top:4px}
  #ablgrid .agc{aspect-ratio:1/1;border-radius:2px;position:relative}
  #ablgrid .agc.cell{cursor:pointer}
  #ablgrid .agc.cell:hover{outline:1px solid var(--accent)}
  #ablgrid .agc.strong{outline:2px solid #ff5d6c;z-index:1}
  #ablgrid .agl{display:flex;align-items:center;justify-content:center;aspect-ratio:auto;
    font-size:8px;font-family:var(--mono);color:var(--muted)}
  .mlnote{font-size:9px;color:var(--muted);font-family:var(--mono);margin-top:6px;line-height:1.5}
  /* policy rows are now clickable (open the microscope) */
  .prow{cursor:pointer;border-radius:6px}
  .prow:hover{background:rgba(110,168,254,.07)}
  .prow.lensed{background:rgba(110,168,254,.12)}
  .prow.lensed .san{color:var(--accent)}

  .status{font-size:12px;color:var(--muted);min-height:16px}
  .status b{color:var(--text)}
  .act{font-size:10px;color:var(--muted);font-family:var(--mono);word-break:break-all}
  .moves{font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.6;
    max-height:70px;overflow-y:auto}

  /* overlays */
  #promo,#loading{position:fixed;inset:0;background:rgba(10,12,16,.72);display:none;
    align-items:center;justify-content:center;z-index:50}
  #promo .box,#loading .box{background:var(--panel);border:1px solid var(--line);
    border-radius:12px;padding:20px;text-align:center}
  #promo .glyphs{display:flex;gap:6px;margin-top:10px}
  #promo .glyphs button{padding:8px 10px;line-height:0}
  #promo .glyphs button img{width:42px;height:42px}
  .spinner{width:26px;height:26px;border:3px solid var(--line);border-top-color:var(--accent);
    border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 12px}
  @keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <div class="left">
    <div>
      <h1>Chessformer (Maia 3) interpretability app</h1>
      <div class="sub" id="modelinfo">loading model…</div>
    </div>
    <div id="boardwrap">
      <div id="board"></div>
      <svg id="arrowsvg" viewBox="0 0 640 640" width="640" height="640"></svg>
    </div>
    <div class="boardctrls">
      <button id="rlbtn" class="medbtn">Residual film</button>
    </div>
  </div>

  <div class="right">
    <div class="card">
      <div class="elorow">
        <h2 style="margin:0">Maia rating (self_elo)</h2>
        <span class="eloval" id="eloval">1500</span>
      </div>
      <input type="range" id="elo" min="600" max="2800" step="25" value="1500">
      <div class="ticks"><span>600</span><span>1100</span><span>1600</span><span>2100</span><span>2800</span></div>
      <label class="lbl cmptoggle"><input type="checkbox" id="cmpchk"> compare with a second rating</label>
      <div id="cmpbox" class="hidden">
        <div class="elorow"><span class="lbl">second rating</span><span class="eloval2" id="cmpval">1100</span></div>
        <input type="range" id="cmpelo" min="600" max="2800" step="25" value="1100">
      </div>
    </div>

    <div class="card">
      <h2>Win / Draw / Loss · side to move</h2>
      <div class="wdl" id="wdl"><div class="w" style="width:33%">—</div><div class="d" style="width:34%"></div><div class="l" style="width:33%"></div></div>
    </div>

    <div class="card" style="flex:1;display:flex;flex-direction:column;min-height:0">
      <h2 id="poltitle">Policy over legal moves</h2>
      <div id="policy"></div>
      <div class="leghint" style="margin-top:5px">click a move to open the move microscope (depth curve + carrier heads)</div>
      <div class="act" id="actfile" style="margin-top:4px"></div>
    </div>

    <div class="card">
      <h2>Moves</h2>
      <div class="moves" id="moves">—</div>
      <div class="fenrow"><input id="fenin" class="fenbox" spellcheck="false" placeholder="paste a FEN to load"><button id="fenload">Load</button></div>
      <div class="controls" style="margin-top:10px">
        <button class="primary" id="newbtn">New game</button>
        <button id="undobtn">← Back</button>
        <select id="color"><option value="white">You play White</option><option value="black">You play Black</option><option value="setup">Set up position</option></select>
        <span class="status" id="status"></span>
      </div>
    </div>
  </div>

  <div class="arch">
    <div class="card" style="flex:1;display:flex;flex-direction:column;min-height:0;overflow:auto">
      <h2>Live attention · this position</h2>
      <div class="attctrls">
        <div class="chiprow"><span class="lbl">Layer</span><div class="chips" id="layerChips"></div></div>
        <div class="chiprow"><span class="lbl">Head</span><div class="chips" id="headChips"></div></div>
        <div class="ablrow"><button id="ablbtn">Ablate this head</button><span class="ablnote">removes its exact residual write</span></div>
      </div>
      <div id="ablout"></div>
      <div class="attcap">Click any square to set the query; hover a heat cell to decompose its geometry below. Top four boards are the selected head; the bottom is the whole layer.</div>
      <div class="attset">
        <div class="attpair">
          <div><div class="attlabel">QKᵀ content logits (semantic):</div><div class="attboard" id="att_qk"></div></div>
          <div><div class="attlabel">GAB bias logits (geometry):</div><div class="attboard" id="att_gab"></div></div>
        </div>
        <div class="attpair">
          <div><div class="attlabel">softmax(QKᵀ) — content only:</div><div class="attboard" id="att_sm_qk"></div></div>
          <div><div class="attlabel">softmax(QKᵀ+GAB) — the head:</div><div class="attboard" id="att_sm_full"></div></div>
        </div>
        <div><div class="attlabel">Whole-layer attention (mean softmax over all heads):</div><div class="attboard" id="att_attn"></div></div>
      </div>
      <div class="attlegend"><span>−</span><div class="legbar div"></div><span>+</span></div>
      <div class="leghint">logit boards: diverging scale, blue = negative · orange = positive (symmetric per board)</div>
      <div class="attlegend"><span>low</span><div class="legbar"></div><span>high</span></div>
      <div class="leghint">softmax boards: attention strength — the content-only / with-GAB pair shares one scale, so what the geometry adds or removes is directly visible</div>

      <div class="smoltitle">Smolgen mixture · how this head's GAB is generated</div>
      <div class="attcap">GAB isn't a stored table — a tiny generator reads this position and emits, per head, 64 mixing coefficients over a static bank of 64×64 square-pair templates shared by every layer &amp; head. The bias above is exactly that weighted sum.</div>
      <div class="gabreadout" id="gabreadout">—</div>
      <div class="attlabel">generated mixing coefficients · template #0–63 · click to inspect</div>
      <div class="coeffstrip" id="coeffstrip"></div>
      <div class="gabdetail hidden" id="gabdetail"></div>
      <div class="attlabel">template vocabulary · the 64 static stencils (row = query sq, col = key sq)</div>
      <div class="gallery" id="gallery"></div>
    </div>
  </div>
</div>

<div id="rlens" class="drawer">
  <div class="mlhead">
    <span class="mltitle">Residual stream across depth · this position</span>
    <span class="mlhint">per-square ‖Δ‖ each structure writes into the stream (viridis) with the logit-lens top move at that point on top — the moving piece sits on its from-square, green ring = destination · side-to-move frame</span>
    <button id="rlclose">✕ close</button>
  </div>
  <div class="residlegend">
    <span class="lg-emb">emb (input)</span>
    <span class="lg-attn">attn add</span>
    <span class="lg-mlp">MLP add</span>
    <span class="lg-enc">enc (final norm — no write, lens only)</span>
  </div>
  <div class="film" id="film"></div>
  <div class="act" id="residinfo" style="margin-top:6px"></div>
</div>

<div id="mlens" class="drawer">
  <div class="mlhead">
    <span class="mltitle" id="mltitle">Move microscope</span>
    <span class="mlhint" id="mlhint">the chosen move's logit at all 18 readout points — where it snaps into the plan — and the heads that carry it</span>
    <button id="mlclose">✕ close</button>
  </div>
  <div class="mlbody">
    <div class="mlchart">
      <div class="attlabel" id="mlchartlab">depth curve · move logit after every sub-layer</div>
      <svg id="mlsvg" viewBox="0 0 660 190"></svg>
    </div>
    <div class="mlgridbox">
      <div class="attlabel">carrier heads · Δlogit = ablated − clean</div>
      <div id="ablgrid"></div>
      <div class="mlnote" id="mlnote"></div>
    </div>
  </div>
</div>

<div id="loading"><div class="box"><div class="spinner"></div><div id="loadtext">Loading Maia3-5M…</div></div></div>
<div id="promo"><div class="box"><div>Promote to</div><div class="glyphs" id="promoglyphs"></div></div></div>

<script>
const FILES=['a','b','c','d','e','f','g','h'];
const PIECE_URI=__PIECE_URI__;   // symbol ('P'..'k') -> svg data URI (pieces.py)
const $=id=>document.getElementById(id);
function pieceImg(sym){ const i=document.createElement('img');
  i.className='pc'; i.src=PIECE_URI[sym]; i.alt=sym; i.draggable=false; return i; }

let API=null, cur=null, orient='white', sel=null, busy=false;
let elo=1500, temp=1, pendingPromo=null, MODEL_INFO=null, setupMode=false;
let cmpOn=false, cmpElo=1100;
const MAXPOL=8;

const sleep=ms=>new Promise(r=>setTimeout(r,ms));

/* ---- wait for the python bridge, then boot (poll; don't rely on the event) ---- */
let booted=false, booting=false, waitN=0;
window.addEventListener('pywebviewready', tryBoot);
function showLoading(msg){
  $('loadtext').textContent=msg;
  $('loading').style.display='flex';
}
function tryBoot(){
  if(booted || booting) return;
  if(window.pywebview && window.pywebview.api){ boot(); return; }
  if(++waitN===1) showLoading('Connecting to Python bridge…');
  if(waitN>120){ showLoading('Bridge not connecting — check the terminal for errors.'); return; }
  setTimeout(tryBoot,100);
}
showLoading('Loading Maia3-5M…');
tryBoot();

async function boot(){
  if(booted || booting) return;
  booting=true;
  showLoading('Loading Maia3-5M…');
  try{
    API = window.pywebview.api;
    let info = await API.info();
    let n=0;
    while(!info.ready && !info.error){ await sleep(400); info = await API.info(); if(++n>300) break; }
    if(info.error){ showLoading('Model failed to load:\n'+info.error); return; }
    if(!info.ready){ showLoading('Model load timed out — check the terminal.'); return; }
    MODEL_INFO = info;
    setModelInfo();
    ensureAttUi(info);
    loadTemplates();   // static GAB vocabulary — fetched once, in the background
    console.log('[maia] bridge ready', info);
    $('loading').style.display='none';
    booted=true;
    await newGame();
  }catch(e){
    showLoading('Bridge error: '+(e && e.message ? e.message : e));
    setTimeout(tryBoot, 500);
  }finally{
    booting=false;
  }
}

function setModelInfo(){
  if(!MODEL_INFO) return;
  const i=MODEL_INFO;
  $('modelinfo').textContent =
    `${i.alias||'Maia3-5M'} · ${i.device||'cpu'} · ${i.num_blocks||8} blocks × ${i.dim_vit||256}d`;
}

/* ---- controls ---- */
$('newbtn').onclick = ()=>{ if(!busy) newGame(); };
$('undobtn').onclick = ()=>{ if(!busy) doUndo(); };
$('color').addEventListener('change', ()=>{ if(busy) return; setupMode=($('color').value==='setup'); if(setupMode) enterSetup(); else newGame(); });
$('fenload').onclick = ()=>{ if(!busy) doSetFen($('fenin').value.trim()); };
async function enterSetup(){
  if(!API || busy) return;
  setupMode=true;
  sel=null;
  cur = await API.analyze();
  renderBoard(); renderMoves();
  await advance();
}
async function doSetFen(fen){
  if(!API || busy || !fen) return;
  sel=null;
  const r = await API.set_fen(fen);
  if(r.error){ setStatus('⚠ '+r.error); return; }
  cur=r; $('color').value='setup'; setupMode=true;
  renderBoard(); renderMoves();
  await advance();
}
async function freeEdit(from, to){
  if(busy) return;
  busy=true;
  cur = await API.edit_square(from, to);
  renderBoard(); renderMoves();
  busy=false;
  await advance();
}
$('elo').addEventListener('input', e=>{
  elo = +e.target.value; $('eloval').textContent = elo;
  scheduleProbe();
});

let probeTimer=null;
function scheduleProbe(){
  clearTimeout(probeTimer);
  probeTimer=setTimeout(probe, 180);   // debounce slider -> re-evaluate same position
}
async function probe(){
  if(!API || busy || !cur || cur.game_over || !cur.human_to_move) return;
  if(cmpOn){
    await refreshCompare();
  } else {
    const d = await API.policy(elo, true);
    if(d.error) return;
    cur = d; renderPolicy(d.policy, d.wdl, d.activation_file, null); renderBoard();
  }
  updateAttention(); updateResidual(); updateMoveLens();
}

async function newGame(){
  sel=null; busy=true; setupMode=false;
  let hc = $('color').value; if(hc==='setup'){ hc='white'; $('color').value='white'; }
  cur = await API.new_game(hc);
  orient = (hc==='black') ? 'black' : 'white';
  renderBoard(); renderMoves(); setModelInfo(); relabelAttCoords();
  busy=false;
  await advance();
}

async function doUndo(){
  if(busy || !API || !cur || !cur.ply) return;
  sel=null;
  cur = await API.undo();
  renderBoard(); renderMoves();
  await advance();
}

/* ---- main loop ---- */
async function advance(){
  // show policy for whoever is to move; if Maia, let it reply
  if(cur.game_over){ finishUI(); return; }
  busy=true;
  let d = await API.policy(elo, true);
  cur = d; renderBoard(); renderPolicy(d.policy, d.wdl, d.activation_file, null);
  setStatus();
  if(d.maia_to_move){
    setStatus(`Maia (${cur.turn}) is thinking…`);
    await sleep(1500);
    const r = await API.maia_move(elo, temp);
    cur = r; renderBoard(); renderMoves();
    renderPolicy(r.maia_policy, r.maia_wdl, r.activation_file, r.maia_move && r.maia_move.uci);
    if(!r.game_over){
      const h = await API.policy(elo, true);
      cur = h; renderBoard(); renderPolicy(h.policy, h.wdl, h.activation_file, null);
    }
  }
  busy=false;
  setStatus();
  if(cmpOn && !cur.game_over && cur.human_to_move) await refreshCompare();
  updateAttention(); updateResidual(); updateMoveLens();
  if(cur.game_over) finishUI();
}

/* ---- board rendering ---- */
function parseFen(fen){
  const map={}; const rows=fen.split(' ')[0].split('/');
  for(let r=0;r<8;r++){ let file=0; for(const ch of rows[r]){
    if(/\d/.test(ch)) file+=+ch;
    else { map[FILES[file]+(8-r)]=ch; file++; }
  }}
  return map;
}
function sqName(row,col){
  return orient==='white' ? FILES[col]+(8-row) : FILES[7-col]+(row+1);
}
function legalTargets(from){
  if(!cur||!cur.legal_moves) return {};
  const t={};
  for(const m of cur.legal_moves) if(m.slice(0,2)===from){ t[m.slice(2,4)]=true; }
  return t;
}
function renderBoard(){
  const board=$('board'); board.innerHTML='';
  const pieces = cur ? parseFen(cur.fen) : {};
  const last = cur && cur.last_move ? [cur.last_move.slice(0,2),cur.last_move.slice(2,4)] : [];
  const targets = (sel && !setupMode) ? legalTargets(sel) : {};
  const hlSq = attQueryReal || null;
  for(let row=0;row<8;row++) for(let col=0;col<8;col++){
    const name=sqName(row,col);
    const fileIdx=FILES.indexOf(name[0]), rankNum=+name[1];
    const isLight=(fileIdx+rankNum)%2===0;
    const d=document.createElement('div');
    d.className='sq '+(isLight?'light':'dark');
    if(last.includes(name)) d.classList.add('lastmove');
    if(sel===name) d.classList.add('sel');
    if(name===hlSq) d.classList.add('attq');
    if(cur && (cur.human_to_move || setupMode)) d.classList.add('playable');
    const pc=pieces[name];
    if(pc){ d.appendChild(pieceImg(pc)); }
    if(targets[name]){
      const dot=document.createElement('span'); dot.className='dot'; d.appendChild(dot);
      if(pc) d.classList.add('cap');
    }
    // edge coordinates
    if(col===0){const c=document.createElement('span');c.className='coord r';c.textContent=name[1];d.appendChild(c);}
    if(row===7){const c=document.createElement('span');c.className='coord f';c.textContent=name[0];d.appendChild(c);}
    d.onclick=()=>onSquare(name);
    board.appendChild(d);
  }
  drawArrow();
}

/* ---- last-move arrow on the SVG overlay above the board ---- */
function sqCenter(name){
  const f=FILES.indexOf(name[0]), r=+name[1];
  const col = orient==='white' ? f : 7-f;
  const row = orient==='white' ? 8-r : r-1;
  return [(col+.5)*80,(row+.5)*80];   // 640px board, 80px squares
}
function drawArrow(){
  const svg=$('arrowsvg'); if(!svg) return;
  svg.innerHTML='';
  if(!cur || !cur.last_move) return;
  const [x1,y1]=sqCenter(cur.last_move.slice(0,2)), [x2,y2]=sqCenter(cur.last_move.slice(2,4));
  const dx=x2-x1, dy=y2-y1, len=Math.hypot(dx,dy); if(len<8) return;
  const ux=dx/len, uy=dy/len, px=-uy, py=ux;
  const head=15, sx=x1+ux*13, sy=y1+uy*13, hx=x2-ux*8, hy=y2-uy*8;
  const bx=hx-ux*head, by=hy-uy*head;
  const ns='http://www.w3.org/2000/svg';
  const g=document.createElementNS(ns,'g');
  g.setAttribute('fill','#f5d56b'); g.setAttribute('opacity','0.75');
  const line=document.createElementNS(ns,'line');
  line.setAttribute('x1',sx); line.setAttribute('y1',sy);
  line.setAttribute('x2',bx); line.setAttribute('y2',by);
  line.setAttribute('stroke','#f5d56b'); line.setAttribute('stroke-width','9');
  line.setAttribute('stroke-linecap','round');
  const tri=document.createElementNS(ns,'polygon');
  tri.setAttribute('points',`${hx},${hy} ${bx+px*9},${by+py*9} ${bx-px*9},${by-py*9}`);
  g.appendChild(line); g.appendChild(tri); svg.appendChild(g);
}

/* ---- click-to-move ---- */
function pieceColorAt(name){
  if(!cur) return null; const pc=parseFen(cur.fen)[name];
  if(!pc) return null; return pc===pc.toUpperCase()?'white':'black';
}
function canMove(color){
  if(!color || !cur) return false;
  return cur.human_color==='both' ? color===cur.turn : color===cur.human_color;
}
function realToCanon(name, turn){
  const file=FILES.indexOf(name[0]); let rank0=(+name[1])-1;
  if(turn==='black') rank0=7-rank0;   // apply the side-to-move board mirror
  return rank0*8+file;
}
async function onSquare(name){
  if(busy || !cur) return;
  if(setupMode){
    if(sel===null){ if(pieceColorAt(name)){ sel=name; renderBoard(); } return; }
    if(name===sel){ await freeEdit(sel, null); sel=null; return; }   // click selected square again = delete
    await freeEdit(sel, name); sel=null; return;
  }
  if(!cur.human_to_move) return;
  if(sel===null){
    if(canMove(pieceColorAt(name)) && legalMovesFrom(name).length){ sel=name; renderBoard(); }
    return;
  }
  if(name===sel){ sel=null; renderBoard(); return; }
  if(canMove(pieceColorAt(name)) && legalMovesFrom(name).length){ sel=name; renderBoard(); return; }
  // attempt sel -> name
  const base=sel+name;
  const promos=cur.legal_moves.filter(m=>m.length>4 && m.slice(0,4)===base);
  if(promos.length){ askPromo(base, promos); return; }
  if(cur.legal_moves.includes(base)){ await doHuman(base); }
  else { sel=null; renderBoard(); }
}
function legalMovesFrom(from){ return cur.legal_moves.filter(m=>m.slice(0,2)===from); }

async function doHuman(uci){
  busy=true; const prev=sel; sel=null;
  const r = await API.human_move(uci);
  if(r.error){ busy=false; setStatus('⚠ '+r.error); sel=prev; renderBoard(); return; }
  cur=r; renderBoard(); renderMoves();
  busy=false;
  await advance();
}

/* promotion picker */
function askPromo(base, promos){
  pendingPromo=base;
  const box=$('promoglyphs'); box.innerHTML='';
  const order=['q','r','b','n'].filter(p=>promos.includes(base+p));
  const white=cur.turn==='white';
  for(const p of order){
    const b=document.createElement('button');
    b.appendChild(pieceImg(white?p.toUpperCase():p));
    b.onclick=async()=>{ $('promo').style.display='none'; const u=pendingPromo+p; pendingPromo=null; await doHuman(u); };
    box.appendChild(b);
  }
  $('promo').style.display='flex';
}

/* ---- panels ---- */
function renderPolicy(pol, wdl, actfile, playedUci){
  const box=$('policy'); box.innerHTML='';
  $('wdl').classList.remove('cmp');
  $('poltitle').textContent = pol ? `Policy over ${pol.length} legal moves` : 'Policy over legal moves';
  if(pol && pol.length){
    pol.slice(0,MAXPOL).forEach((m,i)=>{
      const row=document.createElement('div');
      row.className='prow'+(i===0?' top':'')+(playedUci&&m.uci===playedUci?' played':'')
        +(mlMove&&m.uci===mlMove.uci?' lensed':'');
      row.dataset.uci=m.uci;
      row.onclick=()=>openMoveLens(m.uci, m.san);
      // bar width = the move's actual probability mass (0–100%), so the track reads as a true slider
      row.innerHTML=`<span class="san">${m.san}</span>`+
        `<span class="barwrap"><span class="bar" style="width:${Math.max(1.5,(m.p*100)).toFixed(1)}%"></span></span>`+
        `<span class="pct">${(m.p*100).toFixed(1)}%</span>`;
      box.appendChild(row);
    });
    if(pol.length>MAXPOL){ box.insertAdjacentHTML("beforeend",
      `<div style="font-size:10px;color:var(--muted);margin-top:5px">+${pol.length-MAXPOL} more legal moves</div>`); }
  }
  if(wdl){
    const w=Math.round(wdl.win*100), d=Math.round(wdl.draw*100), l=Math.max(0,100-w-d);
    $('wdl').innerHTML=`<div class="w" style="width:${w}%">${w>8?w+'%':''}</div>`+
      `<div class="d" style="width:${d}%">${d>8?d+'%':''}</div>`+
      `<div class="l" style="width:${l}%">${l>8?l+'%':''}</div>`;
  }
  $('actfile').textContent = actfile ? '↳ saved '+actfile.split('/').slice(-1)[0] : '';
}
function renderMoves(){
  const h=cur && cur.san_history ? cur.san_history : [];
  let out=''; for(let i=0;i<h.length;i+=2){ out+=`${i/2+1}. ${h[i]||''} ${h[i+1]||''}  `; }
  $('moves').textContent = out.trim() || '—';
  const fb=$('fenin'); if(fb && cur && document.activeElement!==fb) fb.value = cur.fen;
}
function setStatus(msg){
  if(msg){ $('status').innerHTML=msg; return; }
  if(cur && cur.game_over){ $('status').innerHTML=`<b>Game over</b> · ${cur.result} (${cur.termination||''})`; return; }
  $('status').textContent='';
}
function finishUI(){ sel=null; renderBoard(); setStatus(); }

/* ---- live attention panel (real QKᵀ / GAB / softmax for the current board) ---- */
let attLayer=0, attHead=0, attQueryReal='d4', lastAtt=null;
let GABT=null, gabtMaxAbs=null;        // static template bank (fetched once) + per-template |max|
let gabTargetReal=null, selTemplate=null;   // decomposition target sq + inspected template
const ATT_IDS=['att_qk','att_gab','att_sm_qk','att_sm_full','att_attn'];

function viridis(t){
  t=Math.max(0,Math.min(1,t));
  const s=[[68,1,84],[59,82,139],[33,144,141],[93,200,99],[253,231,37]];
  const x=t*(s.length-1), i=Math.min(Math.floor(x),s.length-2), f=x-i, a=s[i], b=s[i+1];
  const c=k=>Math.round(a[k]+(b[k]-a[k])*f);
  return `rgb(${c(0)},${c(1)},${c(2)})`;
}
// --- attention colormaps ---
function lerp3(a,b,t){return `rgb(${Math.round(a[0]+(b[0]-a[0])*t)},${Math.round(a[1]+(b[1]-a[1])*t)},${Math.round(a[2]+(b[2]-a[2])*t)})`;}
function divmap(v){ const mid=[24,28,36], blue=[64,132,234], orange=[244,134,58]; return lerp3(mid, v<0?blue:orange, Math.min(1,Math.abs(v))); }

function fillAttCells(el){          // 64 heat cells; click = set query, hover = decompose pair
  for(let idx=0;idx<64;idx++){
    const r=Math.floor(idx/8), c=idx%8, name=sqName(r,c);
    const d=document.createElement('div');
    d.className='attcell '+((FILES.indexOf(name[0])+(+name[1]))%2===0?'lt':'dk');
    d.dataset.idx=idx;
    d.onclick=()=>{ attQueryReal=sqName(Math.floor(idx/8), idx%8); gabTargetReal=null; renderAttention(); renderBoard(); };
    d.onmouseenter=()=>{ gabTargetReal=sqName(Math.floor(idx/8), idx%8); renderGabReadout(); };
    if(c===0){ const sp=document.createElement('span'); sp.className='attcoord r'; sp.textContent=name[1]; d.appendChild(sp); }
    if(r===7){ const sp=document.createElement('span'); sp.className='attcoord f'; sp.textContent=name[0]; d.appendChild(sp); }
    el.appendChild(d);
  }
}
function buildAttBoards(){
  ATT_IDS.forEach(id=>{
    const el=$(id); if(!el || el.children.length) return;
    fillAttCells(el);
  });
}
function relabelAttCoords(){
  ATT_IDS.forEach(id=>{
    const el=$(id); if(!el) return;
    for(const cell of el.children){
      const idx=+cell.dataset.idx, name=sqName(Math.floor(idx/8), idx%8);
      const rc=cell.querySelector('.attcoord.r'); if(rc) rc.textContent=name[1];
      const fc=cell.querySelector('.attcoord.f'); if(fc) fc.textContent=name[0];
    }
  });
}
function paintRow(id, row, colf){
  const el=$(id); if(!el || !row || !cur) return;
  for(const cell of el.children){
    const idx=+cell.dataset.idx, name=sqName(Math.floor(idx/8), idx%8);
    cell.style.background = colf(row[realToCanon(name, cur.turn)]);
    cell.classList.toggle('q', name===attQueryReal);
  }
}
function renderAttention(){
  if(!lastAtt || !cur) return;
  const q = realToCanon(attQueryReal, cur.turn);
  const qk = lastAtt.qk[q], gab = lastAtt.gab[q], attL = (lastAtt.attn_layer||lastAtt.attn)[q];
  if(!qk || !gab || !attL) return;
  // semantic & GAB are pre-softmax logits: diverging scale centered on 0,
  // symmetric per row (blue = negative, orange = positive)
  const dv = row => { let m=0; for(const v of row){ const a=Math.abs(v); if(a>m)m=a; } m=m||1; return v=>divmap(v/m); };
  paintRow('att_qk',  qk,  dv(qk));
  paintRow('att_gab', gab, dv(gab));
  // Side-by-side: what the head would do on content alone vs what it actually
  // does with the geometry added. ONE shared scale so mass shifts are visible;
  // gamma-lift so the secondary squares show, not just the single brightest one.
  // softmax boards are unsigned magnitudes -> viridis (sequential). The signed
  // logit boards above keep the diverging blue/orange map. The content-only and
  // with-GAB pair shares ONE scale so mass shifts are directly comparable;
  // gamma-lift so the secondary squares show, not just the brightest one.
  const attC = lastAtt.attn_content ? lastAtt.attn_content[q] : null;
  const attH = lastAtt.attn ? lastAtt.attn[q] : null;
  if(attC && attH){
    let ms=1e-9; for(const v of attC) if(v>ms) ms=v; for(const v of attH) if(v>ms) ms=v;
    const cf=v=>viridis(Math.pow(v/ms, 0.6));
    paintRow('att_sm_qk',  attC, cf);
    paintRow('att_sm_full', attH, cf);
  }
  // Bottom board = whole-layer mean softmax over heads (still a per-row distribution)
  let mx=1e-9; for(const v of attL) if(v>mx) mx=v;
  paintRow('att_attn', attL, v=>viridis(Math.pow(v/mx, 0.6)));
  renderSmolgen();
}

/* ---- smolgen mixture: coefficients, live decomposition, template vocabulary ---- */
function canonToReal(idx, turn){
  const file=idx%8; let rank0=Math.floor(idx/8);
  if(turn==='black') rank0=7-rank0;   // undo the side-to-move board mirror
  return FILES[file]+(rank0+1);
}
async function loadTemplates(){       // once per app run: the static vocabulary
  if(GABT || !API) return;
  try{
    const d = await API.gab_templates();
    if(!d || d.error){ console.warn('[maia] gab_templates:', d && d.error); return; }
    GABT = d.templates;
    gabtMaxAbs = GABT.map(t=>{ let m=1e-9; for(const row of t) for(const v of row){ const a=Math.abs(v); if(a>m)m=a; } return m; });
    buildGallery();
    renderSmolgen();
  }catch(e){ console.warn('[maia] gab_templates failed', e); }
}
function paintTemplate(cv, t, scale){ // 64×64 pair-matrix -> one pixel per (query, key)
  const ctx=cv.getContext('2d'), img=ctx.createImageData(64,64);
  const mid=[24,28,36], blue=[64,132,234], orange=[244,134,58];
  for(let q=0;q<64;q++) for(let k=0;k<64;k++){
    const v=Math.max(-1,Math.min(1,t[q][k]/scale)), c=v<0?blue:orange, a=Math.abs(v);
    const p=(q*64+k)*4;
    img.data[p]  =Math.round(mid[0]+(c[0]-mid[0])*a);
    img.data[p+1]=Math.round(mid[1]+(c[1]-mid[1])*a);
    img.data[p+2]=Math.round(mid[2]+(c[2]-mid[2])*a);
    img.data[p+3]=255;
  }
  ctx.putImageData(img,0,0);
}
function buildGallery(){
  const g=$('gallery'); if(!g || !GABT || g.children.length) return;
  GABT.forEach((t,i)=>{
    const tile=document.createElement('div'); tile.className='gtile'; tile.dataset.i=i;
    const cv=document.createElement('canvas'); cv.width=64; cv.height=64;
    paintTemplate(cv, t, gabtMaxAbs[i]);
    const lb=document.createElement('div'); lb.className='gtlbl'; lb.innerHTML='<b>#'+i+'</b>';
    tile.appendChild(cv); tile.appendChild(lb);
    tile.onclick=()=>toggleTemplate(i);
    g.appendChild(tile);
  });
}
function toggleTemplate(i){ selTemplate = (selTemplate===i) ? null : i; renderSmolgen(); }
function buildCoeffStrip(n){
  const s=$('coeffstrip'); if(!s || s.children.length===n) return;
  s.innerHTML='';
  for(let i=0;i<n;i++){
    const b=document.createElement('div'); b.className='cbar'; b.dataset.i=i;
    b.appendChild(document.createElement('span'));
    b.onclick=()=>toggleTemplate(i);
    s.appendChild(b);
  }
}
function renderCoeffStrip(){
  const s=$('coeffstrip'); if(!s) return;
  const c=lastAtt && lastAtt.coeffs;
  if(!c){ s.innerHTML=''; return; }
  buildCoeffStrip(c.length);
  let m=1e-9; for(const v of c){ const a=Math.abs(v); if(a>m)m=a; }
  [...s.children].forEach((b,i)=>{
    const v=c[i], sp=b.firstChild, h=Math.max(3, Math.abs(v)/m*48);   // % of strip height (half = 50)
    sp.style.background = v>=0 ? '#f0a35e' : '#6fb3ff';
    sp.style.height=h+'%';
    if(v>=0){ sp.style.bottom='50%'; sp.style.top='auto'; }
    else    { sp.style.top='50%';    sp.style.bottom='auto'; }
    b.title='template #'+i+' · coeff '+v.toFixed(3);
    b.classList.toggle('selt', selTemplate===i);
  });
}
function defaultGabTarget(){          // strongest |GAB| target for the current query
  if(!lastAtt || !cur) return null;
  const row=lastAtt.gab[realToCanon(attQueryReal, cur.turn)]; if(!row) return null;
  let bi=0, bv=-1;
  for(let k=0;k<64;k++){ const a=Math.abs(row[k]); if(a>bv){ bv=a; bi=k; } }
  return canonToReal(bi, cur.turn);
}
function renderGabReadout(){
  const el=$('gabreadout'); if(!el) return;
  if(!lastAtt || !lastAtt.coeffs || !cur){ el.textContent='—'; return; }
  if(!GABT){ el.textContent='loading template bank…'; return; }
  if(!gabTargetReal) gabTargetReal=defaultGabTarget();
  if(!gabTargetReal){ el.textContent='—'; return; }
  const q=realToCanon(attQueryReal, cur.turn), k=realToCanon(gabTargetReal, cur.turn);
  const total=lastAtt.gab[q][k], c=lastAtt.coeffs;
  const terms=c.map((v,i)=>({i, v:v*GABT[i][q][k]}));
  terms.sort((a,b)=>Math.abs(b.v)-Math.abs(a.v));
  const top=terms.slice(0,4);
  let rest=total; for(const t of top) rest-=t.v;
  const f=v=>(v>=0?'+':'−')+Math.abs(v).toFixed(2);
  const cls=v=>v>=0?'pos':'neg';
  let h=`GAB ${attQueryReal}→${gabTargetReal} = <b class="${cls(total)}">${f(total)}</b> &nbsp;=&nbsp; `;
  h+=top.map(t=>`<span class="${cls(t.v)}">${f(t.v)}</span>·<span class="tref" data-i="${t.i}">#${t.i}</span>`).join(' ');
  h+=` <span style="opacity:.6">${f(rest)} rest</span>`;
  el.innerHTML=h;
  el.querySelectorAll('.tref').forEach(x=>{ x.onclick=()=>toggleTemplate(+x.dataset.i); });
}
function renderGalleryBadges(){       // live coefficients on the static vocabulary
  const g=$('gallery'); if(!g || !g.children.length) return;
  const c=lastAtt && lastAtt.coeffs;
  let m=1e-9; if(c) for(const v of c){ const a=Math.abs(v); if(a>m)m=a; }
  [...g.children].forEach((tile,i)=>{
    const lb=tile.querySelector('.gtlbl'), cv=tile.querySelector('canvas');
    if(c && c[i]!==undefined){
      const col=c[i]>=0?'#f0a35e':'#6fb3ff';
      lb.innerHTML=`<b>#${i}</b> <span style="color:${col}">${c[i]>=0?'+':'−'}${Math.abs(c[i]).toFixed(2)}</span>`;
      cv.style.borderColor=divmap(c[i]/m);      // border glows with the live mixture
    } else {
      lb.innerHTML=`<b>#${i}</b>`; cv.style.borderColor='';
    }
    tile.classList.toggle('selt', selTemplate===i);
  });
}
function renderTemplateDetail(){
  const box=$('gabdetail'); if(!box) return;
  if(selTemplate===null || !GABT){ box.classList.add('hidden'); box.innerHTML=''; return; }
  const i=selTemplate, t=GABT[i], c=lastAtt && lastAtt.coeffs ? lastAtt.coeffs[i] : null;
  let rank=null;
  if(c!==null){ rank=1; for(const v of lastAtt.coeffs) if(Math.abs(v)>Math.abs(c)) rank++; }
  box.classList.remove('hidden');
  box.innerHTML=
    `<button class="gdclose" id="gdclose">close</button>`+
    `<div class="gdrow">`+
      `<div><canvas id="gdcv" width="64" height="64"></canvas><div class="gtlbl">full 64×64 · row = query, col = key</div></div>`+
      `<div class="gdboard"><div class="attboard" id="gdrowboard"></div><div class="gtlbl">its row at query ${attQueryReal}</div></div>`+
    `</div>`+
    `<div class="gdinfo">template <b>#${i}</b> — static stencil, shared by every layer &amp; head.`+
    (c===null ? '' :
      ` In L${attLayer}·H${attHead} on this board (elo ${elo}) its coefficient is `+
      `<b>${c>=0?'+':'−'}${Math.abs(c).toFixed(3)}</b> — #${rank} of ${lastAtt.coeffs.length} by |coeff|.`)+
    `</div>`;
  $('gdclose').onclick=()=>toggleTemplate(i);
  paintTemplate($('gdcv'), t, gabtMaxAbs[i]);
  const rb=$('gdrowboard'); fillAttCells(rb);
  if(cur){
    const row=t[realToCanon(attQueryReal, cur.turn)];
    let m=1e-9; for(const v of row){ const a=Math.abs(v); if(a>m)m=a; }
    paintRow('gdrowboard', row, v=>divmap(v/m));
  }
}
function renderSmolgen(){
  renderCoeffStrip();
  renderGabReadout();
  renderGalleryBadges();
  renderTemplateDetail();
}
async function updateAttention(){
  if(!API || !cur || cur.game_over) return;
  ensureAttUi(MODEL_INFO);
  ablExpire();
  try{
    const d = await API.attention(elo, attLayer, attHead);
    if(d && !d.error){
      if(d.num_heads && $('headChips') && $('headChips').children.length !== d.num_heads){
        attHead = Math.min(attHead, d.num_heads - 1);
        buildChips('headChips', d.num_heads, attHead, i=>{ attHead=i; updateAttention(); });
      }
      lastAtt=d; renderAttention();
    }
  }catch(e){ console.warn('[maia] attention update failed', e); }
}

/* ---- exact single-head ablation (engine.ablate_head) ---- */
let lastAblKey=null;
function ablKeyNow(){ return cur ? `${cur.fen}|${elo}|${attLayer}|${attHead}` : null; }
function ablExpire(){ if(lastAblKey && lastAblKey!==ablKeyNow()){ $('ablout').innerHTML=''; lastAblKey=null; } }
const wdlStr=w=>`${Math.round(w.win*100)}/${Math.round(w.draw*100)}/${Math.round(w.loss*100)}`;
$('ablbtn').onclick=async()=>{
  if(!API || !cur || busy || cur.game_over) return;
  const btn=$('ablbtn'); btn.disabled=true; btn.textContent='ablating…';
  try{
    const d=await API.ablate(elo, attLayer, attHead);
    if(!d || d.error){ $('ablout').innerHTML=`<div class="ablhead">⚠ ${d?d.error:'ablation failed'}</div>`; return; }
    lastAblKey=ablKeyNow();
    const rows=d.rows.slice(0,8);
    let h=`<div class="ablhead">block ${d.layer} · head ${d.head} write removed — top ${rows.length} moves by |Δp| · elo ${elo}</div>`;
    for(const r of rows){
      const dlt=(r.p_abl-r.p)*100, cls=dlt>=0?'up':'down';
      h+=`<div class="ablitem"><span class="san">${r.san}</span>`+
         `<span class="vals">${(r.p*100).toFixed(1)} → ${(r.p_abl*100).toFixed(1)}%</span>`+
         `<span class="abldelta ${cls}">${dlt>=0?'+':''}${dlt.toFixed(1)}</span></div>`;
    }
    h+=`<div class="ablwdl">W/D/L ${wdlStr(d.wdl)} → ${wdlStr(d.wdl_abl)}</div>`;
    $('ablout').innerHTML=h;
  }catch(e){
    console.warn('[maia] ablation failed', e);
    $('ablout').innerHTML='<div class="ablhead">⚠ ablation failed — see console</div>';
  }finally{ btn.disabled=false; btn.textContent='Ablate this head'; }
};
function buildChips(id, n, current, onpick){
  const el=$(id); if(!el) return;
  const count = Math.max(0, Number(n) || 0);
  if(!count) return;
  el.innerHTML='';
  for(let i=0;i<count;i++){
    const b=document.createElement('div');
    b.className='chip'+(i===current?' active':''); b.textContent=i; b.dataset.i=i;
    b.onclick=()=>{ el.querySelectorAll('.chip').forEach(c=>c.classList.toggle('active',+c.dataset.i===i)); onpick(i); };
    el.appendChild(b);
  }
}
function ensureAttUi(info){    // idempotent: builds the boards + chips once
  if(!info) return;
  buildAttBoards();
  const lc=$('layerChips'), hc=$('headChips');
  if(lc && !lc.children.length) buildChips('layerChips', info.num_blocks||8, attLayer, i=>{ attLayer=i; updateAttention(); });
  if(hc && !hc.children.length) buildChips('headChips', info.num_heads||8, attHead, i=>{ attHead=i; updateAttention(); });
}

/* ---- residual-stream filmstrip: one combined view in a bottom drawer.
   Each column = one readout point. Heat = per-square ||delta|| the structure
   writes at that point (viridis; emb on its own scale, attn+mlp shared); on
   top, the logit-lens move at that point: piece on from-square, green ring on
   the destination. enc has no additive write, so it shows the lens only. ---- */
let lastRes=null, residOpen=false;
function buildFilm(cols){            // cols: [{label, kind}] — one mini-board each
  const f=$('film'); f.innerHTML='';
  cols.forEach((cd,li)=>{
    const col=document.createElement('div'); col.className='filmcol'+(cd.kind?(' '+cd.kind):'');
    const mb=document.createElement('div'); mb.className='miniboard'; mb.dataset.li=li;
    for(let r=0;r<8;r++) for(let c=0;c<8;c++){
      const d=document.createElement('div'); const sq=(7-r)*8+c;
      d.dataset.sq=sq; d.className=((sq%8+Math.floor(sq/8))%2===1?'lt':'dk');
      mb.appendChild(d);
    }
    const t=document.createElement('div'); t.className='filmlbl'; t.textContent=cd.label;
    col.appendChild(mb); col.appendChild(t); f.appendChild(col);
  });
}
function renderResidual(){
  if(!lastRes) return;
  const film=$('film'), mvs=lastRes.moves, dl=lastRes.delta;   // 18 lens points / 17 writes
  if(film.children.length!==mvs.length) buildFilm(mvs.map(m=>({label:m.label,kind:m.kind})));
  let lo=Infinity,hi=-Infinity,eLo=Infinity,eHi=-Infinity;
  for(const c of dl){ const isE=(c.kind==='emb');
    for(const v of c.norm){ if(isE){ if(v<eLo)eLo=v; if(v>eHi)eHi=v; } else { if(v<lo)lo=v; if(v>hi)hi=v; } } }
  const span=(hi-lo)||1, eSpan=(eHi-eLo)||1;
  [...film.children].forEach((col,li)=>{
    const mv=mvs[li], d=dl[li]||null, cells=col.querySelector('.miniboard').children;
    for(const cell of cells){ const sq=+cell.dataset.sq;
      cell.innerHTML=''; cell.style.boxShadow='';
      if(d){ const isE=(d.kind==='emb');
        cell.style.background=viridis(isE ? (d.norm[sq]-eLo)/eSpan : (d.norm[sq]-lo)/span);
      } else cell.style.background='transparent';                  // enc: lens only
      if(sq===mv.to) cell.style.boxShadow='inset 0 0 0 2px rgba(90,200,120,.95)';
      if(sq===mv.from && mv.piece) cell.appendChild(pieceImg(mv.piece));
    }
    col.querySelector('.filmlbl').textContent = mv.label+(mv.san?' '+mv.san:'');
    col.title = mv.san ? mv.label+' · lens move '+mv.san : mv.label;
  });
  $('residinfo').textContent='heat = ||delta|| the structure writes per square (attn+mlp share one viridis scale, emb scaled on its own) · overlay = logit-lens top legal move of the running stream at that point · elo '+elo;
}
async function updateResidual(){
  if(!API || !cur || cur.game_over || !residOpen) return;
  try{
    const d=await API.residual(elo);
    if(d && !d.error){ lastRes=d; renderResidual(); }
  }catch(e){ console.warn('[maia] residual update failed', e); }
}
function toggleResid(force){
  residOpen = (force!==undefined) ? force : !residOpen;
  if(residOpen && typeof closeMoveLens==='function' && mlMove) closeMoveLens();  // one drawer at a time
  $('rlens').classList.toggle('open', residOpen);
  $('rlbtn').classList.toggle('active', residOpen);
  if(residOpen) updateResidual();
}
$('rlbtn').onclick=()=>toggleResid();
$('rlclose').onclick=()=>toggleResid(false);

/* ---- skill comparison: policy at two ratings, same position ---- */
$('cmpchk').onchange=e=>{
  cmpOn=e.target.checked;
  $('cmpbox').classList.toggle('hidden', !cmpOn);
  scheduleProbe();
};
$('cmpelo').addEventListener('input', e=>{
  cmpElo=+e.target.value; $('cmpval').textContent=cmpElo;
  if(cmpOn) scheduleProbe();
});
async function refreshCompare(){
  if(!API || !cur || cur.game_over) return;
  const d = await API.compare_policy(elo, cmpElo);
  if(!d || d.error) return;
  renderCompare(d);
}
function wdlRowHtml(tag, w){
  const W=Math.round(w.win*100), D=Math.round(w.draw*100), L=Math.max(0,100-W-D);
  return `<div class="wdlrow"><span class="wdltag">${tag}</span>`+
    `<div class="w" style="width:${W}%">${W>12?W+'%':''}</div>`+
    `<div class="d" style="width:${D}%">${D>12?D+'%':''}</div>`+
    `<div class="l" style="width:${L}%">${L>12?L+'%':''}</div></div>`;
}
function renderCompare(d){
  const box=$('policy'); box.innerHTML='';
  $('poltitle').textContent=`Policy · ${d.elo_a} (blue) vs ${d.elo_b} (green)`;
  const rows=d.rows||[];
  rows.slice(0,MAXPOL).forEach(r=>{
    const dlt=(r.p_b-r.p_a)*100, cls=dlt>=0?'up':'down';
    const row=document.createElement('div');
    row.className='prow cmp'+(mlMove&&r.uci===mlMove.uci?' lensed':'');
    row.dataset.uci=r.uci;
    row.onclick=()=>openMoveLens(r.uci, r.san);
    row.title=`${d.elo_a}: ${(r.p_a*100).toFixed(1)}% · ${d.elo_b}: ${(r.p_b*100).toFixed(1)}%`;
    row.innerHTML=`<span class="san">${r.san}</span>`+
      `<span class="dualbar"><span class="bar a" style="width:${Math.max(1,r.p_a*100).toFixed(1)}%"></span>`+
      `<span class="bar b" style="width:${Math.max(1,r.p_b*100).toFixed(1)}%"></span></span>`+
      `<span class="pct delta ${cls}">${dlt>=0?'+':''}${dlt.toFixed(1)}</span>`;
    box.appendChild(row);
  });
  if(rows.length>MAXPOL){ box.insertAdjacentHTML("beforeend",
    `<div style="font-size:10px;color:var(--muted);margin-top:5px">+${rows.length-MAXPOL} more legal moves · Δ = p(${d.elo_b}) − p(${d.elo_a})</div>`); }
  const wdl=$('wdl'); wdl.classList.add('cmp');
  wdl.innerHTML = wdlRowHtml(d.elo_a, d.wdl_a) + wdlRowHtml(d.elo_b, d.wdl_b);
  $('actfile').textContent='';
}

/* ---- move microscope: one move's 18-point depth curve + carrier-head grid ----
   Opens as a drawer from the bottom when a policy row is clicked. Curve = the
   move's logit after every sub-layer (the "snap"); grid = Δlogit from ablating
   every head, sign = ablated − clean (the app-wide convention: what the
   intervention did — negative means the head was supporting the move). */
const KIND_COL={emb:'#8a93a3',attn:'#f0a35e',mlp:'#6fb3ff',enc:'#5ac878'};
let mlMove=null, mlData=null, mlDataB=null, mlGrid=null, mlGridKey=null, mlBusy=false;

function markLensedRows(){
  document.querySelectorAll('#policy .prow').forEach(r=>{
    r.classList.toggle('lensed', !!(mlMove && r.dataset.uci===mlMove.uci));
  });
}
function openMoveLens(uci, san){
  mlMove={uci, san};
  toggleResid(false);                 // one drawer at a time
  $('mlens').classList.add('open');
  markLensedRows();
  updateMoveLens();
}
function closeMoveLens(){ mlMove=null; $('mlens').classList.remove('open'); markLensedRows(); }
$('mlclose').onclick=closeMoveLens;
document.addEventListener('keydown', e=>{
  if(e.key!=='Escape') return;
  if(mlMove) closeMoveLens();
  else if(residOpen) toggleResid(false);
});

async function updateMoveLens(){
  if(!API || !mlMove || !cur) return;
  if(cur.game_over || !cur.legal_moves.includes(mlMove.uci)){ closeMoveLens(); return; }
  if(mlBusy) return;
  mlBusy=true;
  try{
    const d = await API.move_lens(elo, mlMove.uci);
    if(!d || d.error){ console.warn('[maia] move_lens', d && d.error); return; }
    mlData=d;
    mlDataB=null;
    if(cmpOn){
      const b = await API.move_lens(cmpElo, mlMove.uci);
      if(b && !b.error) mlDataB=b;
    }
    drawMlChart();
    const key=`${cur.fen}|${elo}|${mlMove.uci}`;
    if(mlGridKey!==key){
      $('mlnote').textContent='running the 64-head ablation sweep…';
      renderAblGrid(null);
      const g = await API.ablate_grid(elo, mlMove.uci);
      if(g && !g.error){ mlGrid=g; mlGridKey=key; renderAblGrid(g); }
      else $('mlnote').textContent='sweep failed: '+((g && g.error)||'?');
    } else renderAblGrid(mlGrid);
  }catch(e){ console.warn('[maia] move microscope failed', e); }
  finally{ mlBusy=false; }
  markLensedRows();
}

function drawMlChart(){
  const svg=$('mlsvg'); if(!svg || !mlData) return;
  const A=mlData.steps, B=mlDataB ? mlDataB.steps : null;
  const W=660, HH=190, ML=38, MR=12, MT=16, MB=30, iw=W-ML-MR, ih=HH-MT-MB;
  let lo=Infinity, hi=-Infinity;
  for(const s of A){ if(s.logit<lo)lo=s.logit; if(s.logit>hi)hi=s.logit; }
  if(B) for(const s of B){ if(s.logit<lo)lo=s.logit; if(s.logit>hi)hi=s.logit; }
  const pad=(hi-lo)*0.08||1; lo-=pad; hi+=pad;
  const X=i=>ML+iw*i/(A.length-1), Y=v=>MT+ih*(1-(v-lo)/(hi-lo));
  const tip=(s,n,pre)=>`${pre}${s.label} · logit ${s.logit.toFixed(2)} · p ${s.prob!=null?(s.prob*100).toFixed(1)+'%':'—'} · rank ${s.rank!=null?s.rank+'/'+n:'—'}`;
  let h='';
  for(const v of [lo+pad, (lo+hi)/2, hi-pad]){
    h+=`<line x1="${ML}" y1="${Y(v)}" x2="${W-MR}" y2="${Y(v)}" stroke="#262c37"/>`+
       `<text x="${ML-4}" y="${Y(v)+3}" fill="#8b93a3" font-size="9" text-anchor="end" font-family="monospace">${v.toFixed(1)}</text>`;
  }
  // the snap: first point of the final rank-1 run (where the move becomes top for good)
  let snap=-1;
  if(A[A.length-1].rank===1){ snap=A.length-1; while(snap>0 && A[snap-1].rank===1) snap--; }
  if(snap>0){
    h+=`<line x1="${X(snap)}" y1="${MT}" x2="${X(snap)}" y2="${MT+ih}" stroke="#ff5d6c" stroke-dasharray="4 3"/>`+
       `<text x="${Math.min(X(snap)+4, W-110)}" y="${MT+10}" fill="#ff5d6c" font-size="9" font-family="monospace">top from ${A[snap].label}</text>`;
  }
  const line=(S,color,wd)=>`<polyline fill="none" stroke="${color}" stroke-width="${wd}" points="${S.map((s,i)=>X(i)+','+Y(s.logit)).join(' ')}"/>`;
  if(B) h+=line(B,'#7bd88f',1.4);
  h+=line(A,'#6ea8fe',2);
  A.forEach((s,i)=>{ h+=`<circle cx="${X(i)}" cy="${Y(s.logit)}" r="3.2" fill="${KIND_COL[s.kind]||'#fff'}"><title>${tip(s,mlData.n_legal,'')}</title></circle>`; });
  if(B) B.forEach((s,i)=>{ h+=`<circle cx="${X(i)}" cy="${Y(s.logit)}" r="2" fill="#7bd88f" opacity="0.85"><title>${tip(s,mlDataB.n_legal,'elo '+cmpElo+' · ')}</title></circle>`; });
  A.forEach((s,i)=>{
    if(s.kind==='emb'||s.kind==='enc'||s.kind==='mlp'){
      const t = s.kind==='mlp' ? s.label.replace(' mlp','') : s.label;
      h+=`<text x="${X(i)}" y="${HH-10}" fill="#8b93a3" font-size="9" text-anchor="middle" font-family="monospace">${t}</text>`;
    }
  });
  if(B) h+=`<text x="${W-MR}" y="${MT-4}" font-size="9" text-anchor="end" font-family="monospace"><tspan fill="#6ea8fe">━ ${elo}</tspan> <tspan fill="#7bd88f">━ ${cmpElo}</tspan></text>`;
  svg.innerHTML=h;
  $('mltitle').innerHTML=`Move microscope · <b>${mlData.san}</b> <span style="color:var(--muted);font-family:var(--mono);font-size:11px">${mlData.uci} · elo ${elo}${B?' vs '+cmpElo:''}</span>`;
  $('mlchartlab').textContent='depth curve · '+mlData.san+"'s logit after every sub-layer — dots colored by writer (grey emb, orange attn, blue MLP, green final norm); hover for logit / prob / rank";
}

function renderAblGrid(g){
  const el=$('ablgrid'); if(!el) return;
  el.innerHTML='';
  const nb=g ? g.deltas.length : 8, nh=g ? g.deltas[0].length : 8;
  let m=1e-9, sL=-1, sH=-1;
  if(g) g.deltas.forEach((row,L)=>row.forEach((v,hh)=>{ const a=Math.abs(v); if(a>m){ m=a; sL=L; sH=hh; } }));
  el.appendChild(Object.assign(document.createElement('div'),{className:'agc agl'}));
  for(let hh=0;hh<nh;hh++){ const d=document.createElement('div'); d.className='agc agl'; d.textContent='h'+hh; el.appendChild(d); }
  for(let L=0;L<nb;L++){
    const lb=document.createElement('div'); lb.className='agc agl'; lb.textContent='b'+L; el.appendChild(lb);
    for(let hh=0;hh<nh;hh++){
      const d=document.createElement('div'); d.className='agc cell';
      if(g){
        const v=g.deltas[L][hh];
        d.style.background=divmap(v/m);
        d.title=`b${L}·h${hh}  Δ ${v>=0?'+':''}${v.toFixed(2)} — ${v<0?'supports':'suppresses'} ${g.san}`;
        if(L===sL && hh===sH) d.classList.add('strong');
        d.onclick=((L2,H2)=>()=>{             // jump the attention panel to this head
          attLayer=L2; attHead=H2;
          const info=MODEL_INFO||{};
          buildChips('layerChips', info.num_blocks||8, L2, i=>{ attLayer=i; updateAttention(); });
          buildChips('headChips',  info.num_heads ||8, H2, i=>{ attHead=i; updateAttention(); });
          updateAttention();
        })(L,hh);
      } else d.style.background='#10141b';
      el.appendChild(d);
    }
  }
  if(g) $('mlnote').innerHTML=
    `base logit ${g.base_logit.toFixed(2)} · strongest b${sL}·h${sH} `+
    `${g.deltas[sL][sH]>=0?'+':''}${g.deltas[sL][sH].toFixed(2)} · `+
    `blue = ablating the head drops ${g.san}'s logit (carrier) · orange = raises it (suppressor) · `+
    `Δ = ablated − clean · click a cell to open that head in the attention panel`;
}

</script>
</body>
</html>
"""

# Inject the embedded SVG piece set (the only Python in this module).
import json as _json
from pieces import PIECE_URI as _PIECE_URI
INDEX_HTML = INDEX_HTML.replace("__PIECE_URI__", _json.dumps(_PIECE_URI))
