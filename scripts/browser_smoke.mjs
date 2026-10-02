// Run with PLAYWRIGHT_MODULE pointing to an existing Playwright installation.
import path from 'node:path';
import fs from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=process.cwd(), output=path.join(root,'outputs','browser-smoke-'+Date.now());
await fs.mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,args:['--no-sandbox','--proxy-server=direct://']});
const page=await browser.newPage({viewport:{width:1500,height:1000}});
const errors=[];page.on('pageerror',error=>errors.push(error.message));
async function state(){return page.evaluate(async()=>await(await fetch('/api/state')).json());}
async function click(id,route){const [response]=await Promise.all([page.waitForResponse(r=>r.url().endsWith('/api/'+route)&&r.request().method()==='POST'),page.locator('#'+id).click()]);if(!response.ok())throw new Error(await response.text());return response.json();}
async function until(predicate){const deadline=Date.now()+600000;while(Date.now()<deadline){const s=await state();if(predicate(s))return s;await page.waitForTimeout(200);}throw new Error('Timeout waiting for server state');}
async function idle(){const s=await until(s=>!s.busy);if(s.error)throw new Error(s.error);return s;}
try{
 await page.goto('http://127.0.0.1:8765');
 await page.locator('#source').fill(path.join(root,'testdata/phone_short/frames'));
 await page.locator('#mesh').fill(path.join(root,'testdata/phone_short/phone_mm.ply'));
 await page.locator('#directory').fill(path.join(output,'project'));
 await page.locator('#name').fill('Browser acceptance');
 await click('create','create');
 await page.locator('summary').filter({hasText:'Камера'}).click();
 await page.locator('#fx').fill('1867');await page.locator('#fy').fill('1867');await click('camera','settings');
 await page.locator('#posePath').fill(path.join(root,'testdata/phone_short/initial_pose.npy'));
 await click('importPose','pose/import');
 await page.waitForFunction(()=>document.querySelector('#frameImage').naturalWidth>0);
 await click('refine','refine');await idle();
 await page.waitForFunction(()=>!document.querySelector('#accept').disabled);
 await click('accept','accept');
 await page.locator('#rangeEnd').fill('59');
 await click('track','track');
 await until(s=>(s.project.job?.completed.length||0)>=3);
 await click('cancel','cancel');let s=await idle();
 if(s.project.job.status!=='cancelled')throw new Error('Cancel did not persist');
 const completedBefore=s.project.job.completed.length;
 await page.locator('#openDirectory').fill(path.join(output,'project'));
 await click('open','open');
 await page.waitForFunction(()=>!document.querySelector('#resume').disabled);
 await click('resume','resume');s=await idle();
 if(s.project.job.completed.length!==60)throw new Error('Resume did not complete 60 frames');
 const outside=JSON.stringify(s.project.poses['59']);
 await page.locator('#frameNumber').fill('2');await page.locator('#frameNumber').press('Tab');
 await page.locator('#rangeStart').fill('2');await page.locator('#rangeEnd').fill('3');
 await click('track','track');s=await idle();
 if(JSON.stringify(s.project.poses['59'])!==outside)throw new Error('Retrack changed outside range');
 await page.locator('summary').filter({hasText:'Diagnostics и экспорт'}).click();
 const exported=await click('export','export');
 if(exported.files.length!==4)throw new Error('Export missing files');
 await click('preview','preview');s=await idle();
 if(!s.project.last_preview)throw new Error('Preview missing');
 await click('save','save');await click('open','open');
 await page.locator('#frameNumber').fill('45');await page.locator('#frameNumber').press('Tab');
 await page.waitForTimeout(1200);
 await page.screenshot({path:path.join(output,'browser.png'),fullPage:true});
 if(errors.length)throw new Error('Browser errors: '+errors.join('\n'));
 await fs.writeFile(path.join(output,'report.json'),JSON.stringify({passed:true,frames:60,cancel_after:completedBefore,checks:['create','camera','import pose','refine','accept','track','cancel','reopen','resume','range retrack','export','preview','save','reopen'],project:path.join(output,'project'),browser_errors:errors},null,2));
 console.log('PASS '+output);
}finally{await browser.close();}
