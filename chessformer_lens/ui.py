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
<title>Chessformer Interpretability</title>
<style>
  /* Pitch-black neutrals — the app's original scheme, restored 2026-08 after a
     brief lighter run (see interp_plot.py, which still carries the lifted set
     for the paper figures). Mirrored in app.py's window background_color; the
     notebook widgets (interp_widget.py) intentionally stay lighter.
     The lifted set was:
       --bg:#181c24; --panel:#20252f; --panel2:#262c38; --line:#333b4a;
       --chart-bg:#1a1f29; */
  :root{
    --bg:#0e1014; --panel:#161a21; --panel2:#1b2029; --line:#262c37;
    --chart-bg:#0f131a;   /* insets: policy bars, fen box, gab readout, mlsvg */
    --text:#e7eaf0; --muted:#8b93a3; --accent:#6ea8fe; --accent2:#7bd88f;
    --sq-light:#c9d1dc; --sq-dark:#6b7686;
    --hl:rgba(245,213,107,.28); --sel:#7bd88f; --win:#5fb878; --draw:#6b7480; --loss:#d9606a;
    --abl:#ff5d6c;   /* single-head ablation — its overlay on the policy chart */
    --ring-out:#ffffff; --ring-in:#18181b;   /* the query lens, as interp_plot draws it: white halo, dark ring */
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --dock:128px;   /* height of the always-visible peek of the bottom drawer */
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;}
  .wrap{display:flex;gap:18px;padding:16px 20px calc(var(--dock) + 8px);height:100%;align-items:flex-start;overflow:hidden;
    transform-origin:top center;transition:transform .24s ease}
  /* while the microscope is up, the whole top layout shrinks toward the top so the
     drawer can rise 1.5in above the columns' bottom edge without covering anything */
  body.microscope .wrap{transform:scale(var(--shrink,1))}
  .left{display:flex;flex-direction:column;gap:10px}
  .right{flex:1;display:flex;flex-direction:column;gap:12px;min-width:250px;max-width:320px;height:100%}
  .arch{flex:0 0 350px;display:flex;flex-direction:column;height:100%}
  .neur{flex:1 1 214px;min-width:214px;display:flex;flex-direction:column;height:100%}

  h1{font-size:15px;font-weight:600;letter-spacing:.3px;margin:0}
  .sub{font-size:11px;color:var(--muted);font-family:var(--mono)}
  .titlerow{display:flex;align-items:baseline;gap:4px 14px;flex-wrap:wrap}

  /* board — capped at 640px, otherwise a share of the viewport height: what is
     left under it is the microscope drawer's room, so the drawer never has to
     climb over the board (the page never scrolls) */
  #boardwrap{position:relative;width:min(820px, calc(100vh - 330px), calc(100vw - 980px));height:min(820px, calc(100vh - 330px), calc(100vw - 980px))}
  #board{width:100%;height:100%;display:grid;grid-template-columns:repeat(8,1fr);
    grid-template-rows:repeat(8,1fr);border-radius:8px;overflow:hidden;
    box-shadow:0 10px 40px rgba(0,0,0,.45);user-select:none}
  #arrowsvg{position:absolute;inset:0;width:100%;height:100%;z-index:6;pointer-events:none}
  .sq{position:relative;display:flex;align-items:center;justify-content:center;cursor:default}
  .sq.light{background:var(--sq-light)} .sq.dark{background:var(--sq-dark)}
  .sq.lastmove::after{content:"";position:absolute;inset:0;background:var(--hl)}
  .sq.sel{box-shadow:inset 0 0 0 4px var(--sel)}
  /* random-walk replay: each landing square pops as the move goes down */
  @keyframes walkpop{
    0%{box-shadow:inset 0 0 0 4px var(--accent);}
    100%{box-shadow:inset 0 0 0 0 rgba(110,168,254,0);}}
  .sq.walkstep{animation:walkpop .30s ease-out}
  .sq img.pc{position:relative;z-index:2;width:87%;height:87%;
    filter:drop-shadow(0 2px 2px rgba(0,0,0,.30));pointer-events:none}
  .sq .dot{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
    width:30%;height:30%;border-radius:50%;background:rgba(40,50,40,.42);z-index:1;pointer-events:none}
  .sq.cap .dot{width:86%;height:86%;background:transparent;
    box-shadow:inset 0 0 0 4px rgba(40,50,40,.40)}
  .sq.playable{cursor:pointer}
  /* the attention query square: interp_plot's lens — a dark halo around a light ring */
  .sq.attq::before{content:"";position:absolute;inset:12%;border:2.5px solid var(--ring-in);
    box-shadow:0 0 0 3px var(--ring-out), inset 0 0 0 1.5px var(--ring-out);border-radius:3px;z-index:1;pointer-events:none}
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
  /* height grows with the number of rows: 2 ratings, or a clean/ablated pair, or both */
  .wdl.cmp{flex-direction:column;height:auto;gap:2px;font-size:9px}
  .wdl .wdlrow{display:flex;flex:1;min-height:20px;min-width:0;justify-content:flex-start;border-radius:4px;overflow:hidden}
  .wdltag{flex:0 0 30px;display:flex;align-items:center;padding-left:2px;
    color:var(--muted);font-family:var(--mono);background:var(--chart-bg)}

  /* policy list */
  #policy{flex:1;overflow-y:auto;min-height:0}
  .prow{display:grid;grid-template-columns:52px 1fr 52px;align-items:center;gap:10px;
    padding:4px 0;font-size:12px}
  .prow .san{font-family:var(--mono);color:var(--text)}
  .prow .barwrap{display:block;height:18px;background:var(--chart-bg);border:1px solid var(--line);
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
  .dualbar .bar.r{background:var(--abl)}      /* ablated policy */
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
  /* single-head ablation — a toggle; its result is the red overlay on the policy chart */
  .ablrow{display:flex;align-items:center;gap:8px;margin-top:2px}
  #ablbtn{padding:4px 10px;font-size:11px}
  #ablbtn:hover{border-color:var(--abl)}
  #ablbtn.on{background:var(--abl);color:#1a0c0e;border-color:var(--abl);font-weight:700}
  .ablnote{font-size:9px;color:var(--muted);font-family:var(--mono)}
  .ablnote.err{color:var(--loss)}
  .attcap{font-size:10px;color:var(--muted);margin-bottom:10px;line-height:1.45}
  /* neuron panel: layer / index inputs, exact single-unit ablation toggle */
  .nctl{display:flex;gap:6px;align-items:center;flex:1;min-width:0;flex-wrap:wrap}
  .nctl input{width:54px;background:var(--chart-bg);color:var(--text);border:1px solid var(--line);
    border-radius:6px;padding:4px 6px;font-family:var(--mono);font-size:11px}
  .nctl button{padding:3px 8px;font-size:11px}
  #nablbtn:hover{border-color:var(--abl)}
  #nablbtn.on{background:var(--abl);color:#1a0c0e;border-color:var(--abl);font-weight:700}
  /* the network diagram: one column of dots per MLP layer, click a column to pick the
     layer, a dot to pick the unit; the selected layer's dots are its most active units */
  #netsvg{display:block;width:100%;aspect-ratio:240/200;flex:0 0 auto;background:var(--chart-bg);border:1px solid var(--line);border-radius:8px}
  #netsvg .lcol{cursor:pointer}
  #netsvg .unit{cursor:pointer}
  .attset{display:flex;flex-direction:column;gap:12px}
  .attlabel{font-size:10px;color:var(--muted);font-family:var(--mono);margin-bottom:4px;text-align:center;font-weight:600}
  .attboard{width:100%;max-width:202px;aspect-ratio:1/1;margin:0 auto;display:grid;
    grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr);
    border:1px solid var(--line);border-radius:5px;overflow:hidden;background:#10141b}
  .attcell{cursor:pointer;position:relative}
  .attcell.q::before{content:"";position:absolute;inset:12%;border:1.5px solid var(--ring-in);
    box-shadow:0 0 0 2px var(--ring-out), inset 0 0 0 1px var(--ring-out);border-radius:2px;z-index:2;pointer-events:none}
  /* faint checkerboard over the heat so squares stay identifiable */
  .attcell::after{content:"";position:absolute;inset:0;pointer-events:none}
  .attcell.dk::after{background:rgba(0,0,0,.16)}
  .attcell.lt::after{background:rgba(255,255,255,.05)}
  .attcoord{position:absolute;z-index:1;font-size:6px;line-height:1;font-family:var(--mono);
    color:rgba(255,255,255,.9);text-shadow:0 0 2px rgba(0,0,0,.95);pointer-events:none}
  .attcoord.f{right:1px;bottom:0} .attcoord.r{left:1px;top:0}
  .fenrow{display:flex;gap:6px;margin-top:8px}
  .fenbox{flex:1;min-width:0;background:var(--chart-bg);border:1px solid var(--line);border-radius:6px;
    color:var(--text);font-family:var(--mono);font-size:10px;padding:5px 7px}
  #fenload{padding:5px 10px;font-size:11px}
  /* side-by-side board pairs (content vs geometry) */
  .attpair{display:flex;gap:10px}
  .attpair>div{flex:1;min-width:0}
  .attpair .attboard{max-width:none}
  .polhint{font-size:11px;color:var(--muted);margin-top:8px;line-height:1.45}
  .attlegend{display:flex;align-items:center;gap:6px;margin-top:10px;font-size:9px;color:var(--muted);font-family:var(--mono)}
  .legbar{flex:1;height:8px;border-radius:4px;border:1px solid var(--line);
    background:linear-gradient(90deg, rgb(68,1,84), rgb(59,82,139), rgb(33,144,141), rgb(93,200,99), rgb(253,231,37))}
  .legbar.div{background:linear-gradient(90deg, rgb(64,132,234), rgb(24,28,36), rgb(244,134,58))}
  .legbar.pos{background:linear-gradient(90deg, rgb(24,28,36), rgb(244,134,58))}
  .leghint{font-size:8px;color:var(--muted);margin-top:3px;font-family:var(--mono)}



  /* the one bottom-docked drawer — the move microscope. It always peeks up by
     --dock; opened, it rises to the bottom edge of the board — 1.5in above the
     columns' bottom edge — while the layout above scales down to make room (both
     set by fitLayout() from the board's position). */
  .drawer{position:fixed;left:0;right:0;bottom:0;z-index:40;background:var(--panel);
    border:1px solid var(--line);border-bottom:none;border-radius:14px 14px 0 0;
    box-shadow:0 -12px 40px rgba(0,0,0,.45);padding:14px 18px;
    height:var(--drawerh, 300px);display:flex;flex-direction:column;
    transition:transform .24s ease;
    transform:translateY(calc(100% - var(--dock)))}
  .drawer.open{transform:translateY(0);z-index:41;cursor:default}
  /* grip + "pull up" affordance on each peeking window */
  .drawer::before{content:"";position:absolute;top:6px;left:50%;transform:translateX(-50%);
    width:44px;height:4px;border-radius:2px;background:var(--line)}
  .drawer:not(.open){cursor:pointer}
  .drawer:not(.open):hover{background:var(--panel2)}
  .drawer:not(.open)::after{content:"▲ pull up";position:absolute;top:9px;right:14px;
    font-size:9px;font-family:var(--mono);color:var(--muted)}
  /* peek face: title + a clamped hint, no close button */
  .drawer:not(.open) .mlhead{padding-right:64px}   /* room for "▲ pull up" */
  .drawer:not(.open) .mlhead>button{display:none}
  .mlhead{display:flex;align-items:center;gap:8px 14px;margin-bottom:8px;flex-wrap:wrap}
  .mltitle{font-size:19px;font-weight:800;letter-spacing:.2px;white-space:nowrap}
  .mltitle b{color:var(--accent2)}
  .mltags{display:flex;gap:5px;flex-wrap:wrap}
  .mltag{font-size:10px;font-family:var(--mono);color:var(--accent);border:1px solid rgba(110,168,254,.35);
    background:rgba(110,168,254,.08);border-radius:10px;padding:1px 8px;white-space:nowrap}
  .mlhint{font-size:11px;color:var(--muted);font-family:var(--mono);flex:1;min-width:160px}
  #mlclose{padding:3px 10px;font-size:11px}
  /* three sections side by side, each scrolling on its own inside the drawer's height */
  .mlbody{display:flex;gap:22px;align-items:stretch;flex:1;min-height:0}
  .mlchart{flex:1.25;min-width:0;overflow:hidden;display:flex;flex-direction:column}
  #mlsvg{width:100%;height:auto;display:block;background:var(--chart-bg);
    border:1px solid var(--line);border-radius:8px}
  .mlgridbox{flex:0 0 auto;overflow:auto;min-width:300px}
  /* grid-template-columns/rows + width are set per model in renderAblGrid so the
     heatmap has exactly num_heads columns × one row per layer (6/8/16/32 heads). */
  #ablgrid{display:grid;gap:1px;margin-top:4px}
  #ablgrid .agc{aspect-ratio:1/1;border-radius:2px;position:relative}
  #ablgrid .agc.cell{cursor:pointer}
  #ablgrid .agc.cell:hover{outline:1px solid var(--accent)}
  #ablgrid .agc.strong{outline:2px solid #ff5d6c;z-index:1}
  /* final layer: excluded from carrier attribution — dimmed, striped */
  #ablgrid .agc.cell.excl{background:repeating-linear-gradient(45deg,#151a22,#151a22 3px,#1c222c 3px,#1c222c 6px);opacity:.5}
  #ablgrid .agl{display:flex;align-items:center;justify-content:center;aspect-ratio:auto;
    font-size:8px;font-family:var(--mono);color:var(--muted)}
  .mlnote{font-size:9px;color:var(--muted);font-family:var(--mono);margin:2px 0 4px;line-height:1.5}
  /* carrier neurons: one row per unit — name, its per-square footprint, estimate, exact */
  .mlneurons{flex:1;min-width:0;display:flex;flex-direction:column;min-height:0}
  #nrows{flex:1;min-height:0;overflow:auto;margin-top:4px}
  .nrow{display:grid;grid-template-columns:66px 28px 1fr 1fr;align-items:center;gap:8px;
    font-family:var(--mono);font-size:10px;padding:2px 0;border-bottom:1px solid var(--line);cursor:pointer}
  .nrow:hover{background:rgba(110,168,254,.07)}
  .nrow canvas{display:block;width:28px;height:28px;image-rendering:pixelated;border:1px solid var(--line);border-radius:2px;background:#10141b}
  .nrow .neg{color:#6fb3ff} .nrow .pos{color:#f0a35e} .nrow .dim{color:var(--muted)}
  .nrow.strong span:first-child{color:#ff5d6c;font-weight:700}
  /* policy rows are now clickable (open the microscope) */
  .prow{cursor:pointer;border-radius:6px}
  .prow:hover{background:rgba(110,168,254,.07)}
  .prow.lensed{background:rgba(110,168,254,.12)}
  .prow.lensed .san{color:var(--accent)}
  /* compared-move legend in the move microscope */
  .mllegend{display:flex;flex-wrap:wrap;gap:6px;margin:2px 0 6px;min-height:0}
  .mllegend:empty{display:none}
  .mlchip{font-size:11px;font-family:var(--mono);padding:2px 8px;border-radius:10px;cursor:pointer;
    border:1px solid var(--mlc,var(--line));color:var(--mlc,var(--text));background:transparent}
  .mlchip.pri{background:var(--mlc,var(--accent));color:#0a1220;font-weight:600}

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
  .mlh{margin-top:6px} .mlh:empty{display:none}
</style>
</head>
<body>
<div class="wrap">
  <div class="left">
    <div class="titlerow">
      <h1>Chessformer interpretability app</h1>
      <div class="sub" id="modelinfo">loading model…</div>
    </div>
    <div id="boardwrap">
      <div id="board"></div>
      <svg id="arrowsvg" viewBox="0 0 640 640" width="640" height="640"></svg>
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
      <div class="leghint mlh" id="mlh"></div>
    </div>

    <div class="card" style="flex:1;display:flex;flex-direction:column;min-height:0">
      <h2 id="poltitle">Policy over legal moves</h2>
      <div id="policy"></div>
      <div class="leghint" style="margin-top:5px">click a move to analyze it below · up to 4 to compare</div>
      <div class="act" id="actfile" style="margin-top:4px"></div>
    </div>

    <div class="card">
      <h2>Moves</h2>
      <div class="moves" id="moves">—</div>
      <div class="fenrow"><input id="fenin" class="fenbox" spellcheck="false" placeholder="paste a FEN to load"><button id="fenload">Load</button></div>
      <div class="controls" style="margin-top:10px">
        <button class="primary" id="newbtn">New game</button>
        <button id="undobtn">← Back</button>
        <select id="color"><option value="white">You play White</option><option value="black">You play Black</option><option value="setup">Set up position</option><option value="random">Random position</option></select>
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
        <div class="ablrow"><button id="ablbtn">Ablate this head</button><span class="ablnote" id="ablnote">removes its exact residual write — red bars on the policy chart</span></div>
      </div>
      <div class="attcap">Click a square to set the query.</div>
      <div class="attset">
        <div class="attpair">
          <div><div class="attlabel">semantic attention (QKᵀ) </div><div class="attboard" id="att_qk"></div></div>
          <div><div class="attlabel">geometric attention (GAB) </div><div class="attboard" id="att_gab"></div></div>
        </div>
        <div><div class="attlabel">final head attention matrix (scaled softmax(QKᵀ + GAB)) </div><div class="attboard" id="att_sum"></div></div>
        <div>
          <div class="attlegend" style="margin-top:0"><span>0</span><div class="legbar"></div><span>max</span></div>
          <div class="leghint">query's 64 weights (one per key square) sum to 1</div>
        </div>
      </div>
    </div>
  </div>

  <div class="neur">
    <div class="card" style="flex:1;display:flex;flex-direction:column;min-height:0;overflow:auto">
      <h2>Neurons · this position</h2>
      <svg id="netsvg" viewBox="0 0 240 200"></svg>
      <div class="leghint" style="margin-top:5px">columns = MLP layers, top to bottom = unit 0 … N−1; dots = the layer's most active units here, the lit one the selected unit (ringed: causal neurons of the analyzed move). Click a dot, click a height, or ◀ ▶ flip.</div>
      <div class="attctrls" style="margin-top:10px">
        <div class="chiprow"><span class="lbl">Layer</span>
          <div class="nctl"><button id="lprev" title="previous layer">◀</button><input id="nlayer" type="number" min="0" value="0"><button id="lnext" title="next layer">▶</button></div></div>
        <div class="chiprow"><span class="lbl">Unit</span>
          <div class="nctl"><button id="nprev" title="previous unit">◀</button><input id="nidx" type="number" min="0" value="0"><button id="nnext" title="next unit">▶</button></div></div>
        <div class="ablrow"><button id="nablbtn">Ablate this neuron</button></div>
        <div class="ablnote" id="nablnote">removes its exact write on every square — red bars on the policy chart</div>
      </div>
      <div><div class="attlabel" id="nlabel">L0·n0 · activation per square</div><div class="attboard" id="att_neuron"></div></div>
      <div class="attlegend"><span>−</span><div class="legbar div"></div><span>+</span></div>
      <div class="leghint">what this unit fires on, square by square</div>
    </div>
  </div>
</div>

<div id="mlens" class="drawer">
  <div class="mlhead">
    <span class="mltitle" id="mltitle">Analyze one move</span>
    <span class="mltags"><span class="mltag">logit lens</span><span class="mltag">causal heads</span><span class="mltag">causal neurons</span><span class="mltag">residual stream</span></span>
    <span class="mlhint" id="mlhint">click a move in the policy list</span>
    <button id="mlclose">✕ close</button>
  </div>
  <div class="mlbody">
    <div class="mlchart">
      <div class="attlabel">logit lens · the move's logit read off the residual stream at every depth</div>
      <div class="mllegend" id="mllegend"></div>
      <svg id="mlsvg" viewBox="0 0 660 222"></svg>
    </div>
    <div class="mlgridbox" id="mlgridbox">
      <div class="attlabel">causal heads · Δlogit = ablated − clean</div>
      <div class="mlnote" id="mlnote"></div>
      <div id="ablgrid"></div>
    </div>
    <div class="mlneurons" id="mlneurons">
      <div class="attlabel">causal neurons · Δlogit ≈ −∂logit/∂h · h</div>
      <div class="mlnote" id="nnote"></div>
      <div id="nrows"></div>
    </div>
  </div>
</div>

<div id="loading"><div class="box"><div class="spinner"></div><div id="loadtext">Loading model…</div></div></div>
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
const START_FEN='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const EXTRA=144;       // 1.5in at 96dpi: how far the side columns run below the board (fitLayout)
const RAISE=96;        // 1in: how far above the board's bottom edge the opened drawer reaches

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
showLoading('Loading model…');
tryBoot();
fitLayout();

async function boot(){
  if(booted || booting) return;
  booting=true;
  showLoading('Loading model…');
  try{
    API = window.pywebview.api;
    let info = await API.info();
    if(info.target) showLoading('Loading '+info.target+'…');   // name the real model
    let n=0;
    while(!info.ready && !info.error){ await sleep(400); info = await API.info(); if(++n>300) break; }
    if(info.error){ showLoading('Model failed to load:\n'+info.error); return; }
    if(!info.ready){ showLoading('Model load timed out — check the terminal.'); return; }
    MODEL_INFO = info;
    setModelInfo();
    ensureAttUi(info);
    console.log('[maia] bridge ready', info);
    $('loading').style.display='none';
    fitLayout();
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
    `${i.alias||i.target||'Maia3'} · ${i.device||'cpu'} · `+
    `${i.num_blocks||8} layers × ${i.num_heads||8} heads × ${i.dim_vit||256}d`;
  // A model with no rating input (Leela BT4) gets no rating card: the engine
  // ignores the value, so the slider and the second-rating compare would be
  // dead controls. Everything else on the page is derived from the engine.
  const eloCard = $('elo').closest('.card');
  if(eloCard) eloCard.style.display = (i.conditioning === false) ? 'none' : '';
}

/* ---- controls ---- */
$('newbtn').onclick = ()=>{ if(!busy) newGame(); };
$('undobtn').onclick = ()=>{ if(!busy) doUndo(); };
function syncModeSelect(){
  $('color').value = setupMode ? 'setup'
    : (cur && cur.human_color==='black' ? 'black' : 'white');
}
$('color').addEventListener('change', ()=>{
  if(busy){ syncModeSelect(); return; }
  const v = $('color').value;
  if(v==='random') doRandom();        // an action, not a mode — it lands on 'setup'
  else if(v==='setup') enterSetup();
  else if(setupMode) resumeFrom(v);   // leaving set-up: play on from THIS position
  else newGame();                     // white<->black mid-game: start over
});
$('fenload').onclick = ()=>{ if(!busy) doSetFen($('fenin').value.trim()); };
/* A random position is a short Maia self-play walk (bridge.random_position),
   and the walk comes back as frames so we can replay it: the board rockets
   through the game, then eases into the position you are handed. */
const WALK_MS=2000;     // budget for the fast part, whatever the walk's length
async function doRandom(){
  if(!API || busy) return;
  sel=null; busy=true; setupMode=true;
  syncModeSelect();          // the walk leaves you in set-up mode; show that now
  setStatus('rolling a position…');
  let r;
  try{ r = await API.random_position(elo); }
  catch(e){ busy=false; setStatus('⚠ random position failed — see console');
            console.warn('[maia] random_position failed', e); return; }
  if(!r || r.error){ busy=false; setStatus('⚠ '+((r&&r.error)||'random position failed')); return; }
  await playWalk(r);
  cur=r; sel=null; renderBoard(); renderMoves();
  busy=false;
  await advance();
}
async function playWalk(final){
  const fr = final.walk || [];
  if(!fr.length) return;
  const step = Math.min(90, Math.max(26, Math.round(WALK_MS/fr.length)));
  // start from the initial position so the whole game plays out in front of you
  cur = {...final, fen:START_FEN, last_move:null, in_check:false,
         san_history:[], legal_moves:[], game_over:false};
  renderBoard(); renderMoves();
  await sleep(step);
  for(let i=0;i<fr.length;i++){
    const f=fr[i], left=fr.length-1-i;
    cur = {...final, fen:f.fen, last_move:f.last_move, in_check:f.in_check,
           san_history:fr.slice(0,i+1).map(x=>x.san), legal_moves:[], game_over:false};
    renderBoard(); renderMoves(); flashSquare(f.last_move.slice(2,4));
    setStatus(`self-play walk · ${Math.floor(i/2)+1}${i%2?'…':'.'} ${f.san}`);
    // decelerate over the last few plies so the final position registers
    await sleep(left<4 ? step + (4-left)*70 : step);
  }
}
function flashSquare(name){
  const el=$('board').querySelector(`[data-sq="${name}"]`);
  if(!el) return;
  el.classList.remove('walkstep'); void el.offsetWidth; el.classList.add('walkstep');
}
async function enterSetup(){
  if(!API || busy) return;
  setupMode=true;
  sel=null;
  cur = await API.analyze();
  renderBoard(); renderMoves();
  await advance();
}
async function resumeFrom(hc){
  if(!API || busy) return;
  sel=null; busy=true; setupMode=false;
  cur = await API.resume(hc);
  orient = (hc==='black') ? 'black' : 'white';
  renderBoard(); renderMoves(); relabelAttCoords();
  busy=false;
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
    cur = d; renderPolicy(d.policy, d.wdl, d.activation_file, null, d.mlh); renderBoard();
  }
  updateAttention(); updateMoveLens();
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
  cur = d; renderBoard(); renderPolicy(d.policy, d.wdl, d.activation_file, null, d.mlh);
  setStatus();
  if(d.maia_to_move){
    setStatus(`Maia (${cur.turn}) is thinking…`);
    await sleep(1500);
    const r = await API.maia_move(elo, temp);
    cur = r; renderBoard(); renderMoves();
    renderPolicy(r.maia_policy, r.maia_wdl, r.activation_file, r.maia_move && r.maia_move.uci, r.maia_mlh);
    if(!r.game_over){
      const h = await API.policy(elo, true);
      cur = h; renderBoard(); renderPolicy(h.policy, h.wdl, h.activation_file, null, h.mlh);
    }
  }
  busy=false;
  setStatus();
  if(cmpOn && !cur.game_over && cur.human_to_move) await refreshCompare();
  updateAttention(); updateMoveLens();
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
    d.dataset.sq=name;
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
let lastPolArgs=null;         // last renderPolicy call, so overlays can repaint it
function renderPolicy(pol, wdl, actfile, playedUci, mlh){
  lastPolArgs=[pol,wdl,actfile,playedUci,mlh];
  const box=$('policy'); box.innerHTML='';
  const abl=ablPolicy();      // {uci: p_ablated} while the ablation toggle is live
  $('wdl').classList.toggle('cmp', !!abl);
  $('poltitle').textContent = abl
    ? `Policy · clean (blue) vs ${ablData.label} ablated (red)`
    : (pol ? `Policy over ${pol.length} legal moves` : 'Policy over legal moves');
  if(pol && pol.length){
    // ablating re-sorts the list by signed Δ — the moves the head was holding up
    // sink to the bottom, the ones it was suppressing rise to the top
    const list = abl
      ? pol.slice().sort((a,b)=>((abl[b.uci]||0)-b.p)-((abl[a.uci]||0)-a.p))
      : pol;
    list.forEach((m,i)=>{
      const row=document.createElement('div');
      row.className='prow'+(abl?' cmp':(i===0?' top':''))+(playedUci&&m.uci===playedUci?' played':'');
      row.dataset.uci=m.uci;
      row.onclick=()=>openMoveLens(m.uci, m.san);
      if(abl){
        // same shape as the second-rating compare, but the overlay is the ablated pass
        const pb=abl[m.uci]||0, dlt=(pb-m.p)*100, cls=dlt>=0?'up':'down';
        row.title=`clean: ${(m.p*100).toFixed(1)}% · ablated: ${(pb*100).toFixed(1)}%`;
        row.innerHTML=`<span class="san">${m.san}</span>`+
          `<span class="dualbar"><span class="bar a" style="width:${Math.max(1,m.p*100).toFixed(1)}%"></span>`+
          `<span class="bar r" style="width:${Math.max(1,pb*100).toFixed(1)}%"></span></span>`+
          `<span class="pct delta ${cls}">${dlt>=0?'+':''}${dlt.toFixed(1)}</span>`;
      } else {
        // bar width = the move's actual probability mass (0–100%), so the track reads as a true slider
        row.innerHTML=`<span class="san">${m.san}</span>`+
          `<span class="barwrap"><span class="bar" style="width:${Math.max(1.5,(m.p*100)).toFixed(1)}%"></span></span>`+
          `<span class="pct">${(m.p*100).toFixed(1)}%</span>`;
      }
      box.appendChild(row);
    });
    if(abl){ box.insertAdjacentHTML("beforeend",
      `<div style="font-size:10px;color:var(--muted);margin-top:5px">sorted by Δ = p(ablated) − p(clean)</div>`); }
  }
  if(wdl){
    if(abl){
      $('wdl').innerHTML = wdlRowHtml('clean', wdl) + wdlRowHtml('abl', ablData.wdl_abl);
    } else {
      const w=Math.round(wdl.win*100), d=Math.round(wdl.draw*100), l=Math.max(0,100-w-d);
      $('wdl').innerHTML=`<div class="w" style="width:${w}%">${w>8?w+'%':''}</div>`+
        `<div class="d" style="width:${d}%">${d>8?d+'%':''}</div>`+
        `<div class="l" style="width:${l}%">${l>8?l+'%':''}</div>`;
    }
  }
  // moves-left head (Leela): expected plies to the end of the game
  const mlhEl = $('mlh');
  if(mlhEl) mlhEl.textContent = (mlh === null || mlh === undefined) ? '' : `moves-left head: ~${Math.round(mlh)} plies to the end of the game`;
  $('actfile').textContent = actfile ? '↳ saved '+actfile.split('/').slice(-1)[0] : '';
  markLensedRows();
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
const ATT_IDS=['att_qk','att_gab','att_sum','att_neuron'];

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

function fillAttCells(el){          // 64 heat cells; click = set query
  for(let idx=0;idx<64;idx++){
    const r=Math.floor(idx/8), c=idx%8, name=sqName(r,c);
    const d=document.createElement('div');
    d.className='attcell '+((FILES.indexOf(name[0])+(+name[1]))%2===0?'lt':'dk');
    d.dataset.idx=idx;
    d.onclick=()=>{ attQueryReal=sqName(Math.floor(idx/8), idx%8); renderAttention(); renderBoard(); };
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
function paintRow(id, row, colf, markQuery=true){
  const el=$(id); if(!el || !row || !cur) return;
  for(const cell of el.children){
    const idx=+cell.dataset.idx, name=sqName(Math.floor(idx/8), idx%8);
    cell.style.background = colf(row[realToCanon(name, cur.turn)]);
    cell.classList.toggle('q', markQuery && name===attQueryReal);
  }
}
function renderAttention(){
  if(!lastAtt || !cur) return;
  const q = realToCanon(attQueryReal, cur.turn);
  const qk = lastAtt.qk[q], gab = lastAtt.gab[q], attn = lastAtt.attn && lastAtt.attn[q];
  if(!qk || !gab) return;
  // QKᵀ and GAB are pre-softmax logits: diverging scale centered on 0,
  // symmetric per board (blue = negative, orange = positive).
  const dv = row => { let m=0; for(const v of row){ const a=Math.abs(v); if(a>m)m=a; } m=m||1; return v=>divmap(v/m); };
  paintRow('att_qk',  qk,  dv(qk));
  paintRow('att_gab', gab, dv(gab));
  // Third board = the head's ACTUAL attention: softmax(qk + gab) straight from the
  // engine. It's a probability distribution (this query's row sums to 1), so use a
  // sequential viridis scale from 0 to the row's peak weight, not the logit scale.
  if(attn){
    let am=0; for(const v of attn){ if(v>am)am=v; } am=am||1;
    paintRow('att_sum', attn, v=>viridis(v/am));
  } else {
    const sum = qk.map((v,i)=>v+gab[i]);   // fallback: pre-softmax logits
    paintRow('att_sum', sum, dv(sum));
  }
}
function canonToReal(idx, turn){
  const file=idx%8; let rank0=Math.floor(idx/8);
  if(turn==='black') rank0=7-rank0;   // undo the side-to-move board mirror
  return FILES[file]+(rank0+1);
}

async function updateAttention(){
  if(!API || !cur || cur.game_over) return;
  ensureAttUi(MODEL_INFO);
  updateAblation();          // keeps the red policy overlay in step with the picked head
  updateNeuron();            // and the neuron panel (its activation + overlay) with the board
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

/* ---- exact single-head ablation (engine.ablate_head) ----
   A toggle, not a readout: while it is on, the ablated pass is drawn over the
   policy chart in red, exactly the way the second rating is drawn in green.
   The result is cached against fen|elo|layer|head and refetched whenever any of
   those change (board move, elo slider, another head picked). ---- */
let ablOn=false, ablData=null, ablKey=null, ablBusy=false, ablDirty=false;
const ABLNOTE='removes its exact residual write — red bars on the policy chart';
function ablKeyNow(){ return cur ? `${cur.fen}|${elo}|${attLayer}|${attHead}` : null; }
// the overlay the policy chart should draw right now, or null (off / stale / failed).
// One overlay at a time: the head toggle or the neuron toggle, whichever is on.
function ablPolicy(){
  if(!ablData) return null;
  if(ablOn && ablKey===ablKeyNow()) return ablData.p;
  if(nablOn && ablKey===neuronKeyNow()) return ablData.p;
  return null;
}
function ablNote(msg, err){ const n=$('ablnote'); if(!n) return; n.textContent=msg; n.classList.toggle('err', !!err); }
function repaintPolicy(){
  if(cmpOn){ if(lastCmp) renderCompare(lastCmp); }
  else if(lastPolArgs) renderPolicy.apply(null, lastPolArgs);
}
async function updateAblation(){
  const btn=$('ablbtn'); if(!btn) return;
  btn.classList.toggle('on', ablOn);
  btn.textContent = ablOn ? '✓ Ablating this head' : 'Ablate this head';
  if(!ablOn){ ablData=null; ablKey=null; ablNote(ABLNOTE); repaintPolicy(); return; }
  if(!API || !cur || cur.game_over) return;
  const key=ablKeyNow();
  if(ablData && ablKey===key) return;            // cached for this board/elo/head
  if(ablBusy){ ablDirty=true; return; }          // a stale request is in flight
  ablBusy=true; ablNote(`ablating L${attLayer}·h${attHead}…`);
  try{
    const d=await API.ablate(elo, attLayer, attHead);
    if(!d || d.error){
      ablData=null; ablKey=null;
      ablNote('⚠ '+(d && d.error ? d.error : 'ablation failed'), true);
    } else {
      const p={}; for(const r of d.rows) p[r.uci]=r.p_abl;
      ablData={label:`L${d.layer}·h${d.head}`, p:p, wdl_abl:d.wdl_abl}; ablKey=key;
      ablNote(`L${d.layer}·h${d.head} write removed · red = ablated`);
    }
    repaintPolicy();
  }catch(e){
    console.warn('[maia] ablation failed', e);
    ablData=null; ablKey=null; ablNote('⚠ ablation failed — see console', true); repaintPolicy();
  }finally{
    ablBusy=false;
    if(ablDirty){ ablDirty=false; updateAblation(); }
  }
}
$('ablbtn').onclick=()=>{ ablOn=!ablOn; if(ablOn && nablOn){ nablOn=false; updateNeuronAblation(); } updateAblation(); };

/* ---- neuron panel: one MLP unit's activation per square + exact ablation ----
   Mirrors the head panel: pick a unit (inputs, ◀ ▶, or a click in the
   microscope's carrier-neuron table), see what it fires on, toggle its exact
   removal as the red overlay on the policy chart. ---- */
let nLayer=0, nIdx=0, lastNeu=null, nablOn=false, nablBusy=false, nablDirty=false;
let neurOv=null, neurOvKey=null;   // per-layer most-active units, cached per position
const NABLNOTE='removes its exact write on every square — red bars on the policy chart';
function neuronKeyNow(){ return cur ? `${cur.fen}|${elo}|n|${nLayer}|${nIdx}` : null; }
function nablNote(msg, err){ const n=$('nablnote'); if(!n) return; n.textContent=msg; n.classList.toggle('err', !!err); }
function clampNeuron(){
  const i=MODEL_INFO||{}; const nb=i.num_blocks||8, nn=i.mlp_dim||(lastNeu&&lastNeu.n_neurons)||512;
  nLayer=((Math.round(+nLayer)||0)%nb+nb)%nb; nIdx=((Math.round(+nIdx)||0)%nn+nn)%nn;
  $('nlayer').value=nLayer; $('nidx').value=nIdx;
}
function renderNeuron(){
  if(!lastNeu || !cur) return;
  const a=lastNeu.act; let m=1e-9; for(const v of a){ const x=Math.abs(v); if(x>m)m=x; }
  paintRow('att_neuron', a, v=>divmap(v/m), false);
  $('nlabel').textContent=`L${lastNeu.layer}·n${lastNeu.neuron} · activation per square`;
  renderNet();
}
async function updateNeuron(){
  if(!API || !cur || cur.game_over) return;
  clampNeuron();
  const okey=`${cur.fen}|${elo}`;
  if(neurOvKey!==okey){
    try{ const o=await API.neurons_overview(elo); if(o && !o.error){ neurOv=o; neurOvKey=okey; } }
    catch(e){ console.warn('[maia] neurons_overview failed', e); }
  }
  renderNet();
  try{
    const d=await API.neuron(elo, nLayer, nIdx);
    if(d && !d.error){ lastNeu=d; renderNeuron(); }
  }catch(e){ console.warn('[maia] neuron update failed', e); }
  updateNeuronAblation();
}
async function updateNeuronAblation(){
  const btn=$('nablbtn'); if(!btn) return;
  btn.classList.toggle('on', nablOn);
  btn.textContent = nablOn ? '✓ Ablating this neuron' : 'Ablate this neuron';
  if(!nablOn){ if(!ablOn){ ablData=null; ablKey=null; } nablNote(NABLNOTE); repaintPolicy(); return; }
  if(!API || !cur || cur.game_over) return;
  const key=neuronKeyNow();
  if(ablData && ablKey===key) return;
  if(nablBusy){ nablDirty=true; return; }
  nablBusy=true; nablNote(`ablating L${nLayer}·n${nIdx}…`);
  try{
    const d=await API.ablate_neuron(elo, nLayer, nIdx);
    if(!d || d.error){ ablData=null; ablKey=null; nablNote('⚠ '+(d && d.error ? d.error : 'ablation failed'), true); }
    else {
      const p={}; for(const r of d.rows) p[r.uci]=r.p_abl;
      ablData={label:`L${d.layer}·n${d.neuron}`, p:p, wdl_abl:d.wdl_abl}; ablKey=key;
      nablNote(`L${d.layer}·n${d.neuron} write removed · red = ablated`);
    }
    repaintPolicy();
  }catch(e){
    console.warn('[maia] neuron ablation failed', e);
    ablData=null; ablKey=null; nablNote('⚠ ablation failed — see console', true); repaintPolicy();
  }finally{
    nablBusy=false;
    if(nablDirty){ nablDirty=false; updateNeuronAblation(); }
  }
}
$('nablbtn').onclick=()=>{ nablOn=!nablOn; if(nablOn && ablOn){ ablOn=false; updateAblation(); } updateNeuronAblation(); };
$('nlayer').onchange=()=>{ nLayer=+$('nlayer').value; updateNeuron(); };
$('nidx').onchange=()=>{ nIdx=+$('nidx').value; updateNeuron(); };
$('nprev').onclick=()=>{ nIdx-=1; updateNeuron(); };
$('lprev').onclick=()=>{ nLayer-=1; updateNeuron(); };
$('lnext').onclick=()=>{ nLayer+=1; updateNeuron(); };
$('nnext').onclick=()=>{ nIdx+=1; updateNeuron(); };
function buildChips(id, n, current, onpick){
  const el=$(id); if(!el) return;
  const count = Math.max(0, Number(n) || 0);
  if(!count) return;
  // Bare indices — the row's own "Layer"/"Head" label says which is which. The
  // L3·h5 form is still the app-wide name everywhere a head is named on its own
  // (carrier grid, ablation status). (aN/mN name depth points, not heads.)
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

/* ---- the one drawer, and the layout around the board ----
   The microscope is always docked at the bottom as a peek; "open" pulls it up to
   the bottom edge of the board and no further (collapsing keeps its state). The
   right-hand columns are cut to the board's height too, so the policy list ends
   where the board ends and everything below that line belongs to the drawer. */
function fitLayout(){
  const bw=$('boardwrap'), dr=$('mlens'); if(!bw || !dr) return;
  // layout geometry (offset*, unaffected by the open-state scale transform)
  const boardBottom=bw.offsetTop+bw.offsetHeight, colBottom=boardBottom+EXTRA;
  let wrapTop=16;
  document.querySelectorAll('.right, .arch, .neur').forEach(col=>{
    wrapTop=col.offsetTop;
    col.style.height=Math.max(220, colBottom-col.offsetTop)+'px';
  });
  // open, the drawer's top edge sits RAISE above the board's bottom edge; the top
  // layout scales so the columns' bottom edge (and the board's) ends above it
  const dock=parseInt(getComputedStyle(document.documentElement).getPropertyValue('--dock'))||112;
  const drawerTop=boardBottom-RAISE;
  dr.style.height=Math.max(dock+40, window.innerHeight-drawerTop-10)+'px';
  const shrink=Math.min(1, (drawerTop-8-wrapTop)/Math.max(1, colBottom-wrapTop));
  document.documentElement.style.setProperty('--shrink', shrink.toFixed(4));
}
function setDrawerOpen(on){
  $('mlens').classList.toggle('open', !!on);
  document.body.classList.toggle('microscope', !!on);
}
function collapseDrawers(){ setDrawerOpen(false); }
/* clicking the peeking drawer pulls it up; when it's already open, clicks fall
   through to its controls */
$('mlens').addEventListener('click', e=>{
  if($('mlens').classList.contains('open')) return;
  if(e.target.closest('button')) return;
  openMicroscope();
});

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
let lastCmp=null;             // last compare payload, so overlays can repaint it
function renderCompare(d){
  lastCmp=d;
  const box=$('policy'); box.innerHTML='';
  const abl=ablPolicy();      // the ablated pass runs at rating A; drawn as a third bar
  $('poltitle').textContent=`Policy · ${d.elo_a} (blue) vs ${d.elo_b} (green)`+
    (abl?` · ${ablData.label} ablated (red)`:'');
  const rows=d.rows||[];
  rows.forEach(r=>{
    const dlt=(r.p_b-r.p_a)*100, cls=dlt>=0?'up':'down';
    const row=document.createElement('div');
    row.className='prow cmp';
    row.dataset.uci=r.uci;
    row.onclick=()=>openMoveLens(r.uci, r.san);
    const pAbl=abl?(abl[r.uci]||0):0;
    row.title=`${d.elo_a}: ${(r.p_a*100).toFixed(1)}% · ${d.elo_b}: ${(r.p_b*100).toFixed(1)}%`+
      (abl?` · ablated: ${(pAbl*100).toFixed(1)}%`:'');
    row.innerHTML=`<span class="san">${r.san}</span>`+
      `<span class="dualbar"><span class="bar a" style="width:${Math.max(1,r.p_a*100).toFixed(1)}%"></span>`+
      `<span class="bar b" style="width:${Math.max(1,r.p_b*100).toFixed(1)}%"></span>`+
      (abl?`<span class="bar r" style="width:${Math.max(1,pAbl*100).toFixed(1)}%"></span>`:'')+
      `</span>`+
      `<span class="pct delta ${cls}">${dlt>=0?'+':''}${dlt.toFixed(1)}</span>`;
    box.appendChild(row);
  });
  box.insertAdjacentHTML("beforeend",
    `<div style="font-size:10px;color:var(--muted);margin-top:5px">Δ = p(${d.elo_b}) − p(${d.elo_a})</div>`);
  const wdl=$('wdl'); wdl.classList.add('cmp');
  wdl.innerHTML = wdlRowHtml(d.elo_a, d.wdl_a) + wdlRowHtml(d.elo_b, d.wdl_b)
    + (abl?wdlRowHtml('abl', ablData.wdl_abl):'');
  $('actfile').textContent='';
  markLensedRows();
}

/* ---- move microscope: one move's depth curve + carrier-head grid ----
   Opens as a drawer from the bottom when a policy row is clicked. Curve = the
   move's logit after every sub-layer (the "snap"); grid = Δlogit from ablating
   every head, sign = ablated − clean (the app-wide convention: what the
   intervention did — negative means the head was supporting the move). */
const KIND_COL={emb:'#8a93a3',attn:'#f0a35e',mlp:'#6fb3ff',enc:'#5ac878'};
// final layer — excluded from carrier attribution (writes straight to the logits).
// Derived from the loaded model so it tracks the layer count across sizes.
function noCarrierLayer(){ return ((MODEL_INFO && MODEL_INFO.num_blocks) || 8) - 1; }
const MLMAX=4;                                     // how many moves you can overlay at once
const MLCOLORS=['#6ea8fe','#7bd88f','#f0a35e','#c98bff'];
let mlMoves=[];        // [{uci, san, color, steps, n_legal}] — the moves being compared
let mlPrimary=null;    // uci whose carrier-head grid + title are shown
let mlDataB=null;      // elo-B overlay (single-move mode only)
let mlGrid=null, mlGridKey=null, mlBusy=false, mlDirty=false;
let mlNeu=null, mlNeuKey=null;     // carrier-neuron table of the primary move, cached like the grid
let mlTop=null, mlTopKey=null;     // the model's own top move at every readout point (logit lens)

function markLensedRows(){
  const cmap={}; mlMoves.forEach(m=>cmap[m.uci]=m.color);
  document.querySelectorAll('#policy .prow').forEach(r=>{
    const on = r.dataset.uci in cmap;
    r.classList.toggle('lensed', on);
    r.style.boxShadow = on ? ('inset 3px 0 0 '+cmap[r.dataset.uci]) : '';
  });
}
function openMoveLens(uci, san){       // click toggles a move in/out of the comparison
  const i=mlMoves.findIndex(m=>m.uci===uci);
  if(i>=0){                            // already compared -> drop it
    mlMoves.splice(i,1);
    if(!mlMoves.length){ closeMoveLens(); return; }
    if(mlPrimary===uci) mlPrimary=mlMoves[0].uci;
  } else {
    if(mlMoves.length>=MLMAX) return;  // keep the chart readable
    const used=mlMoves.map(m=>m.color);
    const color=MLCOLORS.find(c=>!used.includes(c))||MLCOLORS[mlMoves.length%MLCOLORS.length];
    mlMoves.push({uci, san, color, steps:null, n_legal:0});
    if(!mlPrimary) mlPrimary=uci;
  }
  setDrawerOpen(true);                 // pull the microscope up; the layout above shrinks to make room
  markLensedRows();
  updateMoveLens();
}
function openMicroscope(){              // clicking the microscope's peek (no move needed)
  setDrawerOpen(true);
  if(mlMoves.length) updateMoveLens(); else showMicroscopeEmpty();
}
function showMicroscopeEmpty(){
  $('mltitle').innerHTML='Analyze one move';
  $('mlhint').textContent='click a move in the policy list';
  $('mllegend').innerHTML='';
  $('mlsvg').innerHTML='<text x="330" y="96" fill="#8b93a3" font-size="12" text-anchor="middle" '+
    'font-family="monospace">click a move in the policy list to analyze it</text>';
  $('ablgrid').innerHTML=''; $('mlnote').textContent='';
  $('nrows').innerHTML=''; $('nnote').textContent='';
  mlTop=null; mlTopKey=null;
}
function closeMoveLens(){ mlMoves=[]; mlPrimary=null; mlDataB=null;
  setDrawerOpen(false); markLensedRows(); }
$('mlclose').onclick=closeMoveLens;
document.addEventListener('keydown', e=>{
  if(e.key==='Escape' && $('mlens').classList.contains('open')) closeMoveLens();
});

async function updateMoveLens(){
  if(!API || !mlMoves.length || !cur) return;
  if(mlBusy){ mlDirty=true; return; }   // a click landed mid-fetch -> pick it up when we finish
  mlBusy=true;
  try{
    do{
      mlDirty=false;
      if(cur.game_over){ closeMoveLens(); return; }
      mlMoves = mlMoves.filter(m=>cur.legal_moves.includes(m.uci));   // drop any now-illegal
      if(!mlMoves.length){ closeMoveLens(); return; }
      if(!mlMoves.some(m=>m.uci===mlPrimary)) mlPrimary=mlMoves[0].uci;
      for(const m of mlMoves){                        // a depth curve for each compared move
        const d = await API.move_lens(elo, m.uci);
        if(d && !d.error){ m.steps=d.steps; m.n_legal=d.n_legal; if(d.san) m.san=d.san; }
      }
      mlDataB=null;                                   // elo-B overlay only for a lone move
      if(cmpOn && mlMoves.length===1){
        const b = await API.move_lens(cmpElo, mlMoves[0].uci);
        if(b && !b.error) mlDataB=b;
      }
      const tkey=`${cur.fen}|${elo}`;                  // top move per readout point (one forward)
      if(mlTopKey!==tkey){
        const rd = await API.residual(elo);
        if(rd && !rd.error){ mlTop=rd.moves; mlTopKey=tkey; }
      }
      drawMlChart();
      const pm=mlMoves.find(m=>m.uci===mlPrimary)||mlMoves[0];   // carrier heads = primary move
      const key=`${cur.fen}|${elo}|${pm.uci}`;
      if(mlGridKey!==key){
        $('mlnote').textContent=`running the ${nb_heads()}-head ablation sweep…`;
        renderAblGrid(null);
        const g = await API.ablate_grid(elo, pm.uci);
        if(g && !g.error){ mlGrid=g; mlGridKey=key; renderAblGrid(g); }
        else $('mlnote').textContent='sweep failed: '+((g && g.error)||'?');
      } else renderAblGrid(mlGrid);
      if(mlNeuKey!==key){                                 // carrier neurons = primary move too
        $('nnote').textContent='one backward pass over every MLP unit, then exact checks of the top ones…';
        renderNeurons(null);
        const nd = await API.carrier_neurons(elo, pm.uci);
        if(nd && !nd.error){ mlNeu=nd; mlNeuKey=key; renderNeurons(nd); }
        else $('nnote').textContent='carrier neurons failed: '+((nd && nd.error)||'?');
      } else renderNeurons(mlNeu);
    } while(mlDirty);
  }catch(e){ console.warn('[maia] move microscope failed', e); }
  finally{ mlBusy=false; }
  markLensedRows();
}

function drawMlLegend(series){         // color key for the compared moves; click sets the grid
  const el=$('mllegend'); if(!el) return;
  if(series.length<2){ el.innerHTML=''; return; }
  el.innerHTML=series.map(m=>
    `<span class="mlchip${m.uci===mlPrimary?' pri':''}" data-uci="${m.uci}" style="--mlc:${m.color}">${m.san}</span>`).join('');
  el.querySelectorAll('.mlchip').forEach(c=>{
    c.onclick=()=>{ mlPrimary=c.dataset.uci; markLensedRows(); updateMoveLens(); };
  });
}
function drawMlChart(){
  const svg=$('mlsvg'); if(!svg) return;
  const series=mlMoves.filter(m=>m.steps);
  if(!series.length) return;
  const single=(series.length===1);
  const A=series[0].steps;                          // shared readout grid (x positions + labels)
  const B=(single && mlDataB) ? mlDataB.steps : null;
  // viewBox height follows the chart box's shape, so the chart fills the drawer (text stays unscaled)
  // (SVG elements have no offsetTop — measure with bounding rects; the open
  //  drawer's layout scale is identity, so rects are layout pixels here)
  const box=svg.parentElement, sr=svg.getBoundingClientRect(), brc=box.getBoundingClientRect();
  const bw=sr.width||brc.width||660, avail=brc.bottom-sr.top-4;
  const W=660, HH=Math.round(Math.max(222, Math.min(440, avail>40 ? W*avail/bw : 222)));
  svg.setAttribute('viewBox', `0 0 ${W} ${HH}`);
  const ML=38, MR=12, MT=16, MB=62, iw=W-ML-MR, ih=HH-MT-MB;
  let lo=Infinity, hi=-Infinity;
  for(const m of series) for(const s of m.steps){ if(s.logit<lo)lo=s.logit; if(s.logit>hi)hi=s.logit; }
  if(B) for(const s of B){ if(s.logit<lo)lo=s.logit; if(s.logit>hi)hi=s.logit; }
  const pad=(hi-lo)*0.08||1; lo-=pad; hi+=pad;
  const X=i=>ML+iw*i/(A.length-1), Y=v=>MT+ih*(1-(v-lo)/(hi-lo));
  const tip=(s,n,pre)=>`${pre}${s.label} · logit ${s.logit.toFixed(2)} · p ${s.prob!=null?(s.prob*100).toFixed(1)+'%':'—'} · rank ${s.rank!=null?s.rank+'/'+n:'—'}`;
  let h='';
  for(const v of [lo+pad, (lo+hi)/2, hi-pad]){
    h+=`<line x1="${ML}" y1="${Y(v)}" x2="${W-MR}" y2="${Y(v)}" stroke="#262c37"/>`+
       `<text x="${ML-4}" y="${Y(v)+3}" fill="#8b93a3" font-size="9" text-anchor="end" font-family="monospace">${v.toFixed(1)}</text>`;
  }
  // the snap (first point of the final rank-1 run) only makes sense for a single move
  if(single){
    let snap=-1;
    if(A[A.length-1].rank===1){ snap=A.length-1; while(snap>0 && A[snap-1].rank===1) snap--; }
    if(snap>0){
      h+=`<line x1="${X(snap)}" y1="${MT}" x2="${X(snap)}" y2="${MT+ih}" stroke="#ff5d6c" stroke-dasharray="4 3"/>`+
         `<text x="${Math.min(X(snap)+4, W-110)}" y="${MT+10}" fill="#ff5d6c" font-size="9" font-family="monospace">top from ${A[snap].label}</text>`;
    }
  }
  const line=(S,color,wd)=>`<polyline fill="none" stroke="${color}" stroke-width="${wd}" points="${S.map((s,i)=>X(i)+','+Y(s.logit)).join(' ')}"/>`;
  if(B) h+=line(B,'#7bd88f',1.4);
  for(const m of series) h+=line(m.steps, m.color, m.uci===mlPrimary?2.2:1.5);
  if(single){                                       // writer-colored dots + hover detail
    A.forEach((s,i)=>{ h+=`<circle cx="${X(i)}" cy="${Y(s.logit)}" r="3.2" fill="${KIND_COL[s.kind]||'#fff'}"><title>${tip(s,series[0].n_legal,'')}</title></circle>`; });
    if(B) B.forEach((s,i)=>{ h+=`<circle cx="${X(i)}" cy="${Y(s.logit)}" r="2" fill="#7bd88f" opacity="0.85"><title>${tip(s,mlDataB.n_legal,'elo '+cmpElo+' · ')}</title></circle>`; });
  } else {                                          // one color per move
    for(const m of series) m.steps.forEach((s,i)=>{ h+=`<circle cx="${X(i)}" cy="${Y(s.logit)}" r="2.4" fill="${m.color}"><title>${tip(s,m.n_legal,m.san+' · ')}</title></circle>`; });
  }
  // Depth ticks: emb, aN/mN per layer, enc. Both sub-layers get a tick when they
  // fit; on deep models only the mN (end-of-layer) points are named, so the axis
  // stays readable and still reads one tick per layer.
  const perStep=iw/Math.max(1,A.length-1), tickAll=perStep>=20;
  A.forEach((s,i)=>{
    if(!tickAll && s.kind==='attn') return;
    h+=`<text x="${X(i)}" y="${MT+ih+14}" fill="#8b93a3" font-size="9" text-anchor="middle" font-family="monospace">${s.label}</text>`;
  });
  // the model's own top move at each readout point: one segment per run of the
  // same move, colored like the compared move when it is one of them
  if(mlTop && mlTop.length===A.length){
    const y0=MT+ih+22, sh=14, half=perStep/2, cmap={}; series.forEach(m=>cmap[m.uci]=m.color);
    h+=`<text x="${ML-4}" y="${y0+10}" fill="#8b93a3" font-size="8" text-anchor="end" font-family="monospace">top</text>`;
    let i=0;
    while(i<A.length){
      let j=i; while(j+1<A.length && mlTop[j+1].uci===mlTop[i].uci) j++;
      const x0=Math.max(ML, X(i)-half), x1=Math.min(W-MR, X(j)+half), mv=mlTop[i], col=cmap[mv.uci];
      h+=`<rect x="${x0}" y="${y0}" width="${x1-x0}" height="${sh}" rx="2" fill="${col||'#262c37'}" opacity="${col?0.45:1}">`+
         `<title>${A[i].label}${j>i?'–'+A[j].label:''}: top move ${mv.san||mv.uci||'—'}</title></rect>`;
      const lab=mv.san||mv.uci||'';
      if(lab && (x1-x0) >= lab.length*5.6+4)
        h+=`<text x="${(x0+x1)/2}" y="${y0+10}" fill="${col||'#8b93a3'}" font-size="8.5" text-anchor="middle" font-family="monospace">${lab}</text>`;
      i=j+1;
    }
  }
  if(single && B) h+=`<text x="${W-MR}" y="${MT-4}" font-size="9" text-anchor="end" font-family="monospace"><tspan fill="#6ea8fe">━ ${elo}</tspan> <tspan fill="#7bd88f">━ ${cmpElo}</tspan></text>`;
  svg.innerHTML=h;
  const pm=series.find(m=>m.uci===mlPrimary)||series[0];
  $('mlhint').textContent = series.length>1 ? 'click a move chip to switch the heads and neurons to it' : 'click more policy moves to compare, up to 4';
  $('mltitle').innerHTML=`Analyze one move · <b>${pm.san}</b> <span style="color:var(--muted);font-family:var(--mono);font-size:11px">${(MODEL_INFO&&MODEL_INFO.conditioning===false)?'':'elo '+elo+((single&&B)?' vs '+cmpElo:'')}</span>`;
}

function renderAblGrid(g){
  const el=$('ablgrid'); if(!el) return;
  el.innerHTML='';
  const nb=g ? g.deltas.length : ((MODEL_INFO&&MODEL_INFO.num_blocks)||8),
        nh=g ? g.deltas[0].length : ((MODEL_INFO&&MODEL_INFO.num_heads)||8);
  // Size the grid (num_heads columns × one row per layer) to its third of the
  // drawer: cells as large as possible while the whole grid fits inside the
  // drawer's height (set by fitLayout from the board's position) and a share of
  // the width. Scales across model sizes (6/8/16/32 heads); .mlgridbox scrolls
  // if a huge model still overflows.
  const vw=window.innerWidth||1400, dh=($('mlens')&&$('mlens').clientHeight)||300;
  const maxW=Math.min(vw*0.30, 520), maxH=Math.max(90, dh-132);   // dh less head, labels and the note
  const cs=Math.max(9, Math.floor(Math.min((maxW-16)/nh, (maxH-14)/nb)));
  const gridW=16+nh*cs+nh;  // 16px label col + nh cells + nh 1px gaps
  el.style.gridTemplateColumns = '16px repeat('+nh+','+cs+'px)';
  el.style.gridTemplateRows    = '14px repeat('+nb+','+cs+'px)';
  el.style.width=gridW+'px';
  // Pin the box (and thus #mlnote, which would otherwise ask for its whole
  // sentence on one line and stretch the box wider) to exactly the grid's width.
  const box=$('mlgridbox'); if(box) box.style.width=Math.max(300, gridW)+'px';
  // The final layer writes straight into the logits, so ablating its heads always
  // looks like a huge Δ and drowns out the earlier structure — leave it out of the
  // carrier attribution (color scale + "strongest" pick), just dim it in the grid.
  const NO_CARRIER_LAYER=noCarrierLayer();
  const skip=L=>L===NO_CARRIER_LAYER;
  let m=1e-9, sL=-1, sH=-1;
  if(g) g.deltas.forEach((row,L)=>{ if(skip(L)) return;
    row.forEach((v,hh)=>{ const a=Math.abs(v); if(a>m){ m=a; sL=L; sH=hh; } }); });
  el.appendChild(Object.assign(document.createElement('div'),{className:'agc agl'}));
  // column labels overlap once the cells get small (32 heads in a third of the drawer): then label every 4th
  for(let hh=0;hh<nh;hh++){ const d=document.createElement('div'); d.className='agc agl'; d.textContent=(cs>=16||hh%4===0)?'h'+hh:''; el.appendChild(d); }
  for(let L=0;L<nb;L++){
    const lb=document.createElement('div'); lb.className='agc agl'; lb.textContent='L'+L; el.appendChild(lb);
    for(let hh=0;hh<nh;hh++){
      const d=document.createElement('div'); d.className='agc cell'+(skip(L)?' excl':'');
      if(g){
        const v=g.deltas[L][hh];
        if(skip(L)){
          d.title=`L${L}·h${hh}  Δ ${v>=0?'+':''}${v.toFixed(2)} — excluded from carrier attribution`;
        } else {
          d.style.background=divmap(v/m);
          d.title=`L${L}·h${hh}  Δ ${v>=0?'+':''}${v.toFixed(2)} — ${v<0?'supports':'suppresses'} ${g.san}`;
          if(L===sL && hh===sH) d.classList.add('strong');
        }
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
  if(g && sL>=0) $('mlnote').innerHTML=
    `base logit ${g.base_logit.toFixed(2)} · strongest L${sL}·h${sH} `+
    `${g.deltas[sL][sH]>=0?'+':''}${g.deltas[sL][sH].toFixed(2)} · `+
    `blue = carrier, orange = suppressor · L${NO_CARRIER_LAYER} excluded · click a cell → that head`;
}

function nb_heads(){ const i=MODEL_INFO||{}; return (i.num_blocks||8)*(i.num_heads||8); }

/* ---- carrier neurons: the table under the head grid's convention ----
   One row per unit, strongest first: L·n, an 8×8 footprint of where on the board
   the unit's removal would move the logit (canonical squares mapped back to the
   real board, drawn in the board's orientation), the one-pass estimate and the
   exact re-measurement. Sign as everywhere: blue = carrier, orange = suppressor. */
function renderNeurons(d){
  const el=$('nrows'); if(!el) return;
  el.innerHTML='';
  if(!d || !cur) return;
  let m=1e-9; for(const t of d.top) for(const v of t.squares){ const a=Math.abs(v); if(a>m)m=a; }
  const f=v=>(v>=0?'+':'−')+Math.abs(v).toFixed(2), cls=v=>v<0?'neg':'pos';
  d.top.forEach((t,k)=>{
    const row=document.createElement('div'); row.className='nrow'+(k===0?' strong':'');
    const cv=document.createElement('canvas'); cv.width=8; cv.height=8;
    const ctx=cv.getContext('2d');
    for(let r=0;r<8;r++) for(let c=0;c<8;c++){
      ctx.fillStyle=divmap(t.squares[realToCanon(sqName(r,c), cur.turn)]/m); ctx.fillRect(c,r,1,1);
    }
    const ex=t.exact;
    row.innerHTML=`<span>L${t.layer}·n${t.neuron}</span>`;
    row.appendChild(cv);
    row.insertAdjacentHTML('beforeend',
      `<span class="${cls(t.est)}">Δ ${f(t.est)}</span>`+
      `<span class="${ex==null?'dim':cls(ex)}">${ex==null?'':'exact '+f(ex)}</span>`);
    row.title=`layer ${t.layer} neuron ${t.neuron} · acts most at ${canonToReal(t.peak, cur.turn)} · `+
      `${(ex!=null?ex:t.est)<0?'carries':'suppresses'} ${d.san} · click to open it in the neuron panel`;
    row.onclick=()=>{ nLayer=t.layer; nIdx=t.neuron; updateNeuron(); };
    el.appendChild(row);
  });
  renderNet();
  const NO=noCarrierLayer();
  $('nnote').innerHTML=`base logit ${d.base_logit.toFixed(2)} · ${d.n_layers}×${d.n_neurons} units scored in one backward pass · `+
    `Δ = first-order effect of removing the unit on ${d.san}'s logit · blue = carrier, orange = suppressor · mini-board = where on the board it acts · `+
    `L${NO} excluded, as in the head grid · click a row to open it in the neuron panel`;
}

/* ---- the network diagram: input squares, one column of dots per MLP layer, output.
   The selected layer's dots are its most active units on this position (from
   neurons_overview); other columns show the same count faintly. Carrier units of
   the microscope's primary move get a colored ring. ---- */
function renderNet(){
  const svg=$('netsvg'); if(!svg) return;
  const info=MODEL_INFO||{}; const nb=(neurOv&&neurOv.n_layers)||info.num_blocks||8;
  const N=(neurOv&&neurOv.n_neurons)||info.mlp_dim||512;
  // x: 0–22 index gutter (n0 / nN labels), input squares at 28, layer columns 46–206, outputs at 226
  const W=240, H=200, top=14, bot=172, left=28, right=220, rows=11;
  const colX=i=>46+(206-46)*i/Math.max(1,nb-1);
  const rowY=j=>top+(bot-top)*j/Math.max(1,rows-1);
  const yOf=n=>top+(bot-top)*n/Math.max(1,N-1);      // a unit's height = its index / total
  const carriers={}; if(mlNeu) mlNeu.top.forEach(t=>{ carriers[t.layer+':'+t.neuron]=t; });
  let h='';
  // the net's skeleton: faint evenly spaced dots and links, input squares, output circles
  for(let L=0;L<nb-1;L++) for(let j=0;j<rows;j+=2) for(let k=0;k<rows;k+=4)
    h+=`<line x1="${colX(L)}" y1="${rowY(j)}" x2="${colX(L+1)}" y2="${rowY(k)}" stroke="#262c37" stroke-width="0.6"/>`;
  [1,2,rows-3,rows-2].forEach(j=>{ h+=`<rect x="${left-3}" y="${rowY(j)-3}" width="6" height="6" fill="none" stroke="#8b93a3" stroke-width="0.8"/>`;
    h+=`<line x1="${left+3}" y1="${rowY(j)}" x2="${colX(0)}" y2="${rowY(j)}" stroke="#262c37" stroke-width="0.6"/>`; });
  h+=`<text x="${left}" y="${rowY(Math.floor(rows/2))+3}" fill="#8b93a3" font-size="7" text-anchor="middle" font-family="monospace">⋮</text>`;
  [rows*0.38, rows*0.62].forEach(j=>{ h+=`<circle cx="${right+6}" cy="${rowY(j)}" r="3.5" fill="none" stroke="#8b93a3" stroke-width="0.8"/>`;
    h+=`<line x1="${colX(nb-1)}" y1="${rowY(Math.round(j))}" x2="${right+2}" y2="${rowY(j)}" stroke="#262c37" stroke-width="0.6"/>`; });
  h+=`<text x="${left}" y="${H-8}" fill="#8b93a3" font-size="7" text-anchor="middle" font-family="monospace">in</text>`;
  h+=`<text x="${right+6}" y="${H-8}" fill="#8b93a3" font-size="7" text-anchor="middle" font-family="monospace">out</text>`;
  // the index axis, in its own gutter left of the input squares
  h+=`<text x="2" y="${top+2}" fill="#8b93a3" font-size="6" font-family="monospace">n0</text>`;
  h+=`<text x="2" y="${bot+2}" fill="#8b93a3" font-size="6" font-family="monospace">n${N-1}</text>`;
  h+=`<line x1="9" y1="${top+6}" x2="9" y2="${bot-6}" stroke="#3a4252" stroke-width="0.6" stroke-dasharray="1.5 2"/>`;
  // one column per MLP layer, top = unit 0, bottom = unit N-1
  for(let L=0;L<nb;L++){
    const sel=(L===nLayer), units=(neurOv&&neurOv.layers[L])||[];
    let m=1e-9; for(const u of units) if(u.norm>m) m=u.norm;
    h+=`<g class="lcol" data-l="${L}">`;
    h+=`<rect class="colhit" x="${colX(L)-5}" y="${top-8}" width="10" height="${bot-top+16}" rx="3" fill="${sel?'rgba(110,168,254,.16)':'rgba(0,0,0,0.001)'}"/>`;
    for(let j=0;j<rows;j++) h+=`<circle cx="${colX(L)}" cy="${rowY(j)}" r="2" fill="#262c37"/>`;
    // the layer's most active units on this position, each at its own index's height
    for(const u of units){
      if(sel && u.neuron===nIdx) continue;                 // drawn last, on top
      const car=carriers[L+':'+u.neuron];
      h+=`<circle class="unit" data-l="${L}" data-n="${u.neuron}" cx="${colX(L)}" cy="${yOf(u.neuron)}" r="${sel?3:2.2}" `+
         `fill="${sel?viridis(0.35+0.65*u.norm/m):'#3a4252'}"`+(car?` stroke="${car.est<0?'#6fb3ff':'#f0a35e'}" stroke-width="1.2"`:'')+`>`+
         `<title>L${L} · n${u.neuron} · ‖act‖ ${u.norm.toFixed(1)}${car?' · carrier Δ '+(car.est>=0?'+':'')+car.est.toFixed(2):''}</title></circle>`;
    }
    if(sel){                                               // the lit unit, at index/total
      const u=units.find(x=>x.neuron===nIdx), car=carriers[L+':'+nIdx];
      h+=`<circle class="unit" data-l="${L}" data-n="${nIdx}" cx="${colX(L)}" cy="${yOf(nIdx)}" r="4.2" `+
         `fill="${u?viridis(0.35+0.65*u.norm/m):'#8b93a3'}" stroke="${car?(car.est<0?'#6fb3ff':'#f0a35e'):'#ff5d6c'}" stroke-width="1.6">`+
         `<title>L${L} · n${nIdx}${u?' · ‖act‖ '+u.norm.toFixed(1):''}${car?' · carrier Δ '+(car.est>=0?'+':'')+car.est.toFixed(2):''}</title></circle>`;
    }
    h+=`<text x="${colX(L)}" y="${H-8}" fill="${sel?'#6ea8fe':'#8b93a3'}" font-size="6.5" text-anchor="middle" font-family="monospace">${(nb<=8||L%2===0||sel)?'L'+L:''}</text>`;
    h+=`</g>`;
  }
  svg.innerHTML=h;
  // click a dot = that unit; click anywhere else in a column = the unit at that height
  svg.querySelectorAll('.lcol').forEach(g=>{ g.onclick=e=>{
    const L=+g.dataset.l, dot=e.target.closest('.unit');
    nLayer=L;
    if(dot) nIdx=+dot.dataset.n;
    else {
      const pt=svg.createSVGPoint(); pt.x=e.clientX; pt.y=e.clientY;
      const p=pt.matrixTransform(svg.getScreenCTM().inverse());
      nIdx=Math.round(Math.max(0, Math.min(1, (p.y-top)/(bot-top)))*(N-1));
    }
    updateNeuron();
  }; });
}

/* re-fit the layout and the carrier-head grid to the window */
window.addEventListener('resize', ()=>{
  fitLayout();
  if(mlGrid && $('mlens').classList.contains('open')) renderAblGrid(mlGrid);
  if($('mlens').classList.contains('open')) drawMlChart();
});

</script>
</body>
</html>
"""

# Inject the embedded SVG piece set (the only Python in this module).
import json as _json
from .pieces import PIECE_URI as _PIECE_URI
INDEX_HTML = INDEX_HTML.replace("__PIECE_URI__", _json.dumps(_PIECE_URI))
