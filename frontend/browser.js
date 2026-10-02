// Conventional server-side file browser. Client uploads use native file inputs elsewhere.
export class ServerBrowser {
  constructor(api) {
    this.api=api;this.dialog=document.getElementById('browser');this.history=[];this.cursor=-1;this.recent=JSON.parse(localStorage.getItem('tracker.recentFolders')||'[]');
    this.sort='name';this.ascending=true;
    const $=id=>document.getElementById(id);this.$=$;
    $('browseGo').onclick=()=>this.go($('browsePath').value);
    $('browsePath').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();this.go(e.target.value);}};
    $('browseName').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();this.choose();}};
    $('browseBack').onclick=()=>{if(this.cursor>0)this.go(this.history[--this.cursor],false);};
    $('browseForward').onclick=()=>{if(this.cursor+1<this.history.length)this.go(this.history[++this.cursor],false);};
    $('browseUp').onclick=()=>this.go(this.data.parent);
    $('browseSelect').onclick=()=>this.choose();$('browseFilter').onchange=()=>this.table();
    document.querySelectorAll('[data-sort]').forEach(th=>th.onclick=()=>{this.ascending=this.sort===th.dataset.sort?!this.ascending:true;this.sort=th.dataset.sort;this.table();});
    this.dialog.addEventListener('close',()=>{if(this.resolve){this.resolve(null);this.resolve=null;}});
  }
  async open({title='Browse Server',folder=false,extensions=[],project=false,start='' }={}) {
    this.options={folder,extensions,project};this.$('browseTitle').textContent=title+(folder?' — выбор папки':'');this.$('browseSelect').textContent=folder?'Select Folder':'Open';
    this.$('browseFilter').replaceChildren(new Option(folder?'Folders':project?'project.json':extensions.join(', ')||'All Files','context'),new Option('All Files','all'));
    this.history=[];this.cursor=-1;this.$('browserError').textContent='';this.$('browseName').value='';
    return new Promise(resolve=>{this.resolve=resolve;this.dialog.showModal();this.go(start||this.recent[0]||'');});
  }
  async go(path,history=true) {
    let success=false;const requestId=this.requestId=(this.requestId||0)+1;this.$('browsePath').disabled=true;this.$('browseGo').disabled=true;this.$('browseSelect').disabled=true;
    try {const data=await this.api('browse?path='+encodeURIComponent(path));if(requestId!==this.requestId)return;success=true;this.data=data;this.$('browsePath').value=data.path;this.$('browseName').value='';this.selected=null;this.$('browserError').textContent='';
      if(history){this.history=this.history.slice(0,this.cursor+1);this.history.push(data.path);this.cursor=this.history.length-1;}
      this.$('browseBack').disabled=this.cursor<=0;this.$('browseForward').disabled=this.cursor>=this.history.length-1;
      this.$('breadcrumbs').replaceChildren();let current='';const parts=data.path.split('/').filter(Boolean);['/',...parts].forEach(part=>{current=part==='/'?'/':current.replace(/\/$/,'')+'/'+part;const target=current,b=document.createElement('button');b.textContent=part+' ›';b.onclick=()=>this.go(target);this.$('breadcrumbs').appendChild(b);});
      this.$('places').replaceChildren();for(const item of [...data.places,...this.recent.slice(0,5).map(path=>({name:'Recent · '+(path.split('/').pop()||'/'),path}))]){const b=document.createElement('button');b.textContent=item.name;b.title=item.path;b.onclick=()=>this.go(item.path);this.$('places').appendChild(b);}this.table();
    }catch(e){this.$('browserError').textContent=e.message;}finally{if(requestId===this.requestId){this.$('browsePath').disabled=false;this.$('browseGo').disabled=false;this.$('browseSelect').disabled=!success;}}
  }
  table(){if(!this.data)return;const all=this.$('browseFilter').value==='all';const files=this.data.entries.filter(e=>e.directory||all||(!this.options.folder&&(this.options.project?e.name==='project.json':!this.options.extensions.length||this.options.extensions.some(ext=>e.name.toLowerCase().endsWith(ext)))));
    files.sort((a,b)=>{if(a.directory!==b.directory)return a.directory?-1:1;const av=a[this.sort]??'',bv=b[this.sort]??'';return (typeof av==='number'?av-bv:String(av).localeCompare(String(bv),undefined,{numeric:true}))*(this.ascending?1:-1);});
    const body=this.$('browseEntries');body.replaceChildren();for(const item of files){const tr=document.createElement('tr');tr.dataset.path=item.path;for(const text of [(item.directory?'▸ ':'')+item.name,item.type,item.size==null?'—':formatBytes(item.size),new Date(item.modified*1000).toLocaleString()]){const td=document.createElement('td');td.textContent=text;tr.appendChild(td);}tr.onclick=()=>{body.querySelectorAll('tr').forEach(r=>r.classList.remove('selected'));tr.classList.add('selected');this.selected=item;this.$('browseName').value=item.name;};tr.ondblclick=()=>{if(item.directory)this.go(item.path);else{tr.click();this.choose();}};body.appendChild(tr);}}
  choose(){if(!this.data)return;const name=this.$('browseName').value.trim();let selected=this.selected;if(name&&name!==selected?.name)selected=this.data.entries.find(e=>e.name===name);if(name&&!selected){this.$('browserError').textContent='Выберите существующий файл или папку';return;}
    if(this.options.folder&&selected&&!selected.directory){this.$('browserError').textContent='Нужно выбрать папку';return;}
    if(!this.options.folder&&(!selected||selected.directory)){if(selected)this.go(selected.path);return;}
    const path=selected?.path||this.data.path;this.recent=[this.data.path,...this.recent.filter(p=>p!==this.data.path)].slice(0,8);localStorage.setItem('tracker.recentFolders',JSON.stringify(this.recent));const resolve=this.resolve;this.resolve=null;this.dialog.close();resolve(path);
  }
}
export function formatBytes(n){if(n<1024)return n+' B';if(n<1024**2)return(n/1024).toFixed(1)+' KB';if(n<1024**3)return(n/1024**2).toFixed(1)+' MB';return(n/1024**3).toFixed(2)+' GB';}
