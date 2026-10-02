import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const source=await fs.readFile(new URL('../frontend/pose-math.js',import.meta.url),'utf8');
const M=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);
const m=M.fromValues([0,0,90,10,20,700]);
const local=M.translate(m,0,5,'local'),global=M.translate(m,0,5,'global');
close(local[0][3],10);close(local[1][3],25);close(global[0][3],15);close(global[1][3],20);
for(const space of ['local','global']){const r=M.rotate(m,0,Math.PI/2,space);const R=r.slice(0,3).map(row=>row.slice(0,3));for(let i=0;i<3;i++)for(let j=0;j<3;j++)close(M.dot(R[i],R[j]),+(i===j));assert.deepEqual(M.origin(r),M.origin(m));}
assert.notDeepEqual(M.rotate(m,0,.5,'local'),M.rotate(m,0,.5,'global'));
for(const v of [[20,30,40,1,2,300],[-60,80,170,0,-10,900]]){const result=M.fromValues(M.toValues(M.fromValues(v)));M.fromValues(v).forEach((row,i)=>row.forEach((n,j)=>close(n,result[i][j])));}
const c={fx:1000,fy:1000,cx:960,cy:540};
close(M.axisParameter(M.ray([1010,540],c),[0,0,700],[1,0,0]),35);
assert.equal(M.axisParameter([0,0,1],[0,0,700],[0,0,1]),null);
console.log('PASS: Local/Global rigid transforms, Euler convention and perspective axis constraint');
