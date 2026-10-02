// Canonical object -> OpenCV camera pose; all translation and geometry are mm.
export const identity=()=>[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
export const copy=m=>m.map(row=>[...row]);
export const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
export const add=(a,b)=>a.map((v,i)=>v+b[i]);
export const sub=(a,b)=>a.map((v,i)=>v-b[i]);
export const scale=(a,s)=>a.map(v=>v*s);
export const norm=a=>Math.sqrt(dot(a,a));
export const unit=a=>scale(a,1/(norm(a)||1));
export const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export function multiply(a,b){return a.map(row=>b[0].map((_,j)=>row.reduce((v,x,k)=>v+x*b[k][j],0)));}
export function fromValues([rx,ry,rz,tx,ty,tz]){[rx,ry,rz]=[rx,ry,rz].map(x=>x*Math.PI/180);const cx=Math.cos(rx),sx=Math.sin(rx),cy=Math.cos(ry),sy=Math.sin(ry),cz=Math.cos(rz),sz=Math.sin(rz);return [[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx,tx],[sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx,ty],[-sy,cy*sx,cy*cx,tz],[0,0,0,1]];}
export function toValues(m){const y=Math.asin(Math.max(-1,Math.min(1,-m[2][0])));const x=Math.abs(Math.cos(y))>1e-7?Math.atan2(m[2][1],m[2][2]):Math.atan2(-m[1][2],m[1][1]);const z=Math.abs(Math.cos(y))>1e-7?Math.atan2(m[1][0],m[0][0]):0;return [x*180/Math.PI,y*180/Math.PI,z*180/Math.PI,m[0][3],m[1][3],m[2][3]];}
export const origin=m=>m.slice(0,3).map(row=>row[3]);
export const axis=(m,i,space)=>space==='local'?m.slice(0,3).map(row=>row[i]):[0,1,2].map(j=>+(i===j));
export function translate(m,i,amount,space){const out=copy(m),v=axis(m,i,space);for(let j=0;j<3;j++)out[j][3]+=v[j]*amount;return out;}
export function rotate(m,i,angle,space){const v=[0,0,0];v[i]=1;const c=Math.cos(angle),s=Math.sin(angle),t=1-c;const[x,y,z]=v;const r=[[t*x*x+c,t*x*y-s*z,t*x*z+s*y],[t*x*y+s*z,t*y*y+c,t*y*z-s*x],[t*x*z-s*y,t*y*z+s*x,t*z*z+c]];const R=m.slice(0,3).map(row=>row.slice(0,3));const updated=space==='local'?multiply(R,r):multiply(r,R);const out=copy(m);updated.forEach((row,j)=>row.forEach((value,k)=>out[j][k]=value));return out;}
export function transform(m,p){return m.slice(0,3).map(row=>dot(row.slice(0,3),p)+row[3]);}
export function project(p,c){return p[2]>1?[c.fx*p[0]/p[2]+c.cx,c.fy*p[1]/p[2]+c.cy]:null;}
export function ray(uv,c){return unit([(uv[0]-c.cx)/c.fx,(uv[1]-c.cy)/c.fy,1]);}
export function axisParameter(r,p,a){const c=dot(a,r),den=1-c*c;return den<1e-5?null:(c*dot(r,p)-dot(a,p))/den;}
export function planeVector(r,p,normal){const denominator=dot(r,normal);if(Math.abs(denominator)<1e-6)return null;const t=dot(p,normal)/denominator;if(t<=0)return null;const vector=sub(scale(r,t),p);return norm(vector)>1e-7?unit(vector):null;}
export function signedAngle(a,b,normal){return Math.atan2(dot(normal,cross(a,b)),dot(a,b));}
