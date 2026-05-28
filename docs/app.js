
// hero neural-net animation
(function(){
  const c=document.getElementById('net'); if(!c) return;
  const x=c.getContext('2d'); let W,H,nodes=[],edges=[],pulses=[];
  const layers=[4,6,6,5,3];
  function build(){
    W=c.width=c.offsetWidth*devicePixelRatio; H=c.height=c.offsetHeight*devicePixelRatio;
    nodes=[]; edges=[];
    const pad=W*0.08, gw=(W-pad*2)/(layers.length-1);
    layers.forEach((cnt,li)=>{
      for(let i=0;i<cnt;i++){
        const gy=H/(cnt+1)*(i+1);
        nodes.push({x:pad+gw*li,y:gy,l:li,p:Math.random()*Math.PI*2});
      }
    });
    for(let a=0;a<nodes.length;a++)for(let b=0;b<nodes.length;b++)
      if(nodes[b].l===nodes[a].l+1) edges.push([a,b]);
  }
  function spawn(){
    if(pulses.length<26 && edges.length){
      const e=edges[(Math.random()*edges.length)|0];
      pulses.push({a:e[0],b:e[1],t:0,s:0.012+Math.random()*0.02});
    }
  }
  function draw(){
    x.clearRect(0,0,W,H);
    x.lineWidth=1*devicePixelRatio;
    edges.forEach(e=>{const A=nodes[e[0]],B=nodes[e[1]];
      x.strokeStyle='rgba(255,255,255,0.05)';x.beginPath();x.moveTo(A.x,A.y);x.lineTo(B.x,B.y);x.stroke();});
    pulses.forEach(p=>{const A=nodes[p.a],B=nodes[p.b];p.t+=p.s;
      const px=A.x+(B.x-A.x)*p.t, py=A.y+(B.y-A.y)*p.t;
      const g=x.createRadialGradient(px,py,0,px,py,7*devicePixelRatio);
      g.addColorStop(0,'rgba(94,240,192,0.9)');g.addColorStop(1,'rgba(94,240,192,0)');
      x.fillStyle=g;x.beginPath();x.arc(px,py,7*devicePixelRatio,0,7);x.fill();});
    pulses=pulses.filter(p=>p.t<1);
    const t=Date.now()/900;
    nodes.forEach(n=>{const r=(2.4+Math.sin(t+n.p)*0.9)*devicePixelRatio;
      x.fillStyle='rgba(94,240,192,'+(0.5+Math.sin(t+n.p)*0.3)+')';
      x.beginPath();x.arc(n.x,n.y,r,0,7);x.fill();});
    if(Math.random()<0.32) spawn();
    requestAnimationFrame(draw);
  }
  build(); draw(); addEventListener('resize',build);
})();
