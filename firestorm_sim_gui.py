import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch, Circle as mplCircle

# ----------------- Core simulation -----------------

def simulate_firestorm(ignites, hitboxes, area_frac, duration, trials,
                       storm_r=5.6, bolt_r=1.0, imp_r=1.8, freq=10.0,
                       cov_samples=1000):
    """Run Monte‑Carlo fire‑storm simulation once and return
    means ± SEM for hits, plus coverage stats and last frame data."""

    ignites   = int(ignites)
    trials    = int(trials)
    scheduled = int(duration * freq)
    imp_count = min(ignites * 5, scheduled)
    ord_count = scheduled - imp_count

    scale = np.sqrt(1 + area_frac)
    s_r   = storm_r * scale
    b_r   = bolt_r  * scale
    i_r   = imp_r   * scale

    # sample points for coverage integration
    theta_s = np.random.rand(cov_samples) * 2*np.pi
    r_s     = np.sqrt(np.random.rand(cov_samples)) * s_r
    xs_s, ys_s = r_s * np.cos(theta_s), r_s * np.sin(theta_s)

    # per‑trial storage
    hits_ord = {r: [] for r in hitboxes}
    hits_imp = {r: [] for r in hitboxes}
    cov_ord_trials, cov_imp_trials = [], []
    last_positions = last_cov_ord = last_cov_imp = None

    for t in range(trials):
        # bolt locations
        th_o = np.random.rand(ord_count)*2*np.pi
        r_o  = np.sqrt(np.random.rand(ord_count))*s_r
        xo, yo = r_o*np.cos(th_o), r_o*np.sin(th_o)
        th_i = np.random.rand(imp_count)*2*np.pi
        r_i  = np.sqrt(np.random.rand(imp_count))*s_r
        xi, yi = r_i*np.cos(th_i), r_i*np.sin(th_i)

        # coverage fractions this trial
        d_o = np.hypot(xs_s[:,None]-xo, ys_s[:,None]-yo)
        d_i = np.hypot(xs_s[:,None]-xi, ys_s[:,None]-yi)
        cov_o_mask = np.any(d_o <= b_r, axis=1)
        cov_i_mask = np.any(d_i <= i_r, axis=1)
        cov_ord_trials.append(cov_o_mask.mean())
        cov_imp_trials.append(cov_i_mask.mean())

        if t == trials-1:
            last_positions = (xo, yo, xi, yi)
            last_cov_ord, last_cov_imp = cov_o_mask, cov_i_mask

        # hit counts vs each radius
        dist_o = np.hypot(xo, yo)
        dist_i = np.hypot(xi, yi)
        for rad in hitboxes:
            hits_ord[rad].append(np.sum(dist_o <= rad + b_r))
            hits_imp[rad].append(np.sum(dist_i <= rad + i_r))

    # aggregate stats (mean & SEM)
    avg, sem = {}, {}
    for rad in hitboxes:
        arr_o = np.array(hits_ord[rad], dtype=float)
        arr_i = np.array(hits_imp[rad], dtype=float)
        avg[rad] = {'ordinary': arr_o.mean(), 'improved': arr_i.mean()}
        sem[rad] = {'ordinary': arr_o.std(ddof=1)/np.sqrt(trials),
                    'improved': arr_i.std(ddof=1)/np.sqrt(trials)}

    cov_arr_o = np.array(cov_ord_trials)
    cov_arr_i = np.array(cov_imp_trials)
    cov_avg_ord = cov_arr_o.mean(); cov_sem_ord = cov_arr_o.std(ddof=1)/np.sqrt(trials)
    cov_avg_imp = cov_arr_i.mean(); cov_sem_imp = cov_arr_i.std(ddof=1)/np.sqrt(trials)

    return (avg, sem,
            last_positions,
            (s_r, b_r, i_r),
            (cov_avg_ord, cov_avg_imp, cov_sem_ord, cov_sem_imp),
            (xs_s, ys_s, last_cov_ord, last_cov_imp))

# ----------------- Drawing helpers -----------------

def draw_sample(ax, xo, yo, xi, yi, storm_r, bolt_r, imp_r, radii,
                samp_x, samp_y, cov_ord_mask, cov_imp_mask):
    """Plot a single sample storm with hitboxes and coverage shading."""
    ax.clear(); ax.set_aspect('equal')
    max_r = storm_r + imp_r
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')

    storm_c, imp_c, ord_c = 'black', 'red', 'darkorange'
    cmap = plt.get_cmap('tab20b').colors

    # coverage shading
    ax.scatter(samp_x[cov_imp_mask], samp_y[cov_imp_mask], s=5, color=imp_c, alpha=0.3)
    ax.scatter(samp_x[cov_ord_mask], samp_y[cov_ord_mask], s=5, color=ord_c, alpha=0.2)

    # storm & bolts
    ax.add_patch(mplCircle((0,0), storm_r, fill=False, lw=2, edgecolor=storm_c))
    for xb,yb in zip(xi, yi): ax.add_patch(mplCircle((xb,yb), imp_r, fill=False, edgecolor=imp_c, alpha=0.6))
    for xb,yb in zip(xo, yo): ax.add_patch(mplCircle((xb,yb), bolt_r, fill=False, linestyle='--', edgecolor=ord_c, alpha=0.6))

    # hitboxes & legend
    elems=[Patch(edgecolor=storm_c, facecolor='none', label='Storm')]
    for idx, rad in enumerate(radii):
        col=cmap[idx%len(cmap)]
        elems.append(Patch(edgecolor=col, facecolor=col, alpha=0.3, label=f'Hitbox {rad:.2f}m'))
        theta=np.random.rand()*2*np.pi; rpos=np.sqrt(np.random.rand())*(storm_r-rad)
        ax.add_patch(mplCircle((rpos*np.cos(theta), rpos*np.sin(theta)), rad, fill=True, facecolor=col, alpha=0.3, edgecolor=col))
    elems.extend([Patch(edgecolor=imp_c, facecolor='none', label='Improved bolt'),
                  Patch(edgecolor=ord_c, facecolor='none', linestyle='--', label='Ordinary bolt')])
    ax.legend(handles=elems, loc='center left', bbox_to_anchor=(1.02,0.5), fontsize='small')

    # coverage label
    ax.text(0.95,0.95,f'Coverage:\nImp {cov_imp_mask.mean()*100:.1f}%\nOrd {cov_ord_mask.mean()*100:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize='small',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    ax.set_title('Sample Trial')

# ----------------- GUI -----------------

def build_gui():
    root=tk.Tk(); root.title('Firestorm Simulator')
    root.protocol('WM_DELETE_WINDOW', lambda:(root.destroy(),sys.exit(0)))

    # ---- controls ----
    ctrl = ttk.Frame(root, padding=10); ctrl.grid(row=0,column=0,sticky='nsew')
    # sliders
    ttk.Label(ctrl,text='Ignites consumed').grid(row=0,column=0,sticky='e'); ign_s=tk.Scale(ctrl,from_=0,to=12,orient='horizontal'); ign_s.set(3); ign_s.grid(row=0,column=1,sticky='w')
    ttk.Label(ctrl,text='Area mod (%)').grid(row=1,column=0,sticky='e'); area_s=tk.Scale(ctrl,from_=-90,to=100,orient='horizontal'); area_s.set(0); area_s.grid(row=1,column=1,sticky='w')
    ttk.Label(ctrl,text='Duration (s)').grid(row=2,column=0,sticky='e'); dur_s=tk.Scale(ctrl,from_=1,to=12,orient='horizontal'); dur_s.set(6); dur_s.grid(row=2,column=1,sticky='w')
    # entries
    ttk.Label(ctrl,text='Enemy hitbox radii (m)').grid(row=3,column=0,sticky='e'); ent_h=ttk.Entry(ctrl,width=25); ent_h.insert(0,'0.5,1.0'); ent_h.grid(row=3,column=1,sticky='w')
    ttk.Label(ctrl,text='Trials').grid(row=4,column=0,sticky='e'); ent_t=ttk.Entry(ctrl,width=10); ent_t.insert(0,'1000'); ent_t.grid(row=4,column=1,sticky='w')
    # run buttons
    btn_sim=ttk.Button(ctrl,text='Run simulation'); btn_sim.grid(row=5,column=0,columnspan=2,pady=4)
    # scan
    ttk.Label(ctrl,text='Scan variable').grid(row=6,column=0,sticky='e'); scan_cb=ttk.Combobox(ctrl,values=['Ignites consumed','Area mod (%)','Duration (s)'],state='readonly',width=18); scan_cb.current(0); scan_cb.grid(row=6,column=1,sticky='w')
    btn_scan=ttk.Button(ctrl,text='Run scan'); btn_scan.grid(row=7,column=0,columnspan=2,pady=4)

    # ---- output ----
    out=ttk.Frame(root); out.grid(row=0,column=1,sticky='nsew')
    fig,axarr=plt.subplots(1,2,figsize=(10,5)); fig.subplots_adjust(right=0.85)
    canvas=FigureCanvasTkAgg(fig,master=out); canvas.get_tk_widget().pack(fill='both',expand=True)
    txt=tk.Text(out,height=6); txt.pack(fill='x')

    # ---- callbacks ----
    def run_sim():
        try:
            ign=ign_s.get(); area=area_s.get()/100; dur=dur_s.get(); hrs=[float(x) for x in ent_h.get().split(',')]; tr=int(ent_t.get())
        except: messagebox.showerror('Input error','Bad inputs'); return
        txt.delete('1.0','end'); fig.suptitle('')
        avg, sem, lp, radii_scaled, cov_stats, cov_last = simulate_firestorm(ign,{r:1 for r in hrs},area,dur,tr)
        xs,ys,cov_o,cov_i=cov_last; xo,yo,xi,yi=lp; s_r,b_r,i_r=radii_scaled
        cov_avg_o,cov_avg_i,cov_sem_o,cov_sem_i=cov_stats
        ax0=axarr[0]; ax0.clear()
        radii_sorted=sorted(avg); x=np.arange(len(radii_sorted))
        ord_mean=[avg[r]['ordinary'] for r in radii_sorted]; imp_mean=[avg[r]['improved'] for r in radii_sorted]
        ord_err=[sem[r]['ordinary'] for r in radii_sorted]; imp_err=[sem[r]['improved'] for r in radii_sorted]
        w=0.35
        ax0.bar(x-w/2,ord_mean,w,yerr=ord_err,capsize=4,label='Ord'); ax0.bar(x+w/2,imp_mean,w,yerr=imp_err,capsize=4,label='Imp')
        ax0.set_xticks(x); ax0.set_xticklabels([f'{r:.2f}m' for r in radii_sorted]); ax0.set_ylabel('Avg hits'); ax0.legend()
        for r in radii_sorted:
            txt.insert('end',f'{r:.2f}m -> Ord {avg[r]['ordinary']:.2f}±{sem[r]['ordinary']:.2f}, Imp {avg[r]['improved']:.2f}±{sem[r]['improved']:.2f}\n')
        txt.insert('end',f'Coverage Ord {cov_avg_o*100:.1f}±{cov_sem_o*100:.1f}%, Imp {cov_avg_i*100:.1f}±{cov_sem_i*100:.1f}%\n')
        draw_sample(axarr[1],xo,yo,xi,yi,s_r,b_r,i_r,radii_sorted,xs,ys,cov_o,cov_i); canvas.draw()

    def run_scan():
        try:
            ign=ign_s.get(); area=area_s.get()/100; dur=dur_s.get(); hrs=[float(x) for x in ent_h.get().split(',')]; tr=int(ent_t.get())
        except: messagebox.showerror('Input error','Bad inputs'); return
        txt.delete('1.0','end'); fig.suptitle('')
        var=scan_cb.get(); steps=11
        if var=='Ignites consumed': scan_vals=np.linspace(0,12,steps)
        elif var=='Area mod (%)': scan_vals=np.linspace(-90,100,steps)
        else: scan_vals=np.linspace(1,12,steps)
        ord_m, imp_m, ord_e, imp_e=[],[],[],[]; rad=hrs[0]
        for val in scan_vals:
            ign_i=val if var=='Ignites consumed' else ign
            area_i=val/100 if var=='Area mod (%)' else area
            dur_i=val if var=='Duration (s)' else dur
            avg, sem, *_ = simulate_firestorm(int(ign_i),{rad:1},area_i,dur_i,tr)
            ord_m.append(avg[rad]['ordinary']); imp_m.append(avg[rad]['improved'])
            ord_e.append(sem[rad]['ordinary']); imp_e.append(sem[rad]['improved'])
        # output scatterplot data to text box
        txt.insert('end', 'value,Ord_mean,Ord_err,Imp_mean,Imp_err')
        for i, v in enumerate(scan_vals):
            txt.insert('end', f"{v:.3f},{ord_m[i]:.3f},{ord_e[i]:.3f},{imp_m[i]:.3f},{imp_e[i]:.3f}")
        ax0=axarr[0]; ax0.clear()
        ax0.errorbar(scan_vals,ord_m,yerr=ord_e,fmt='o-',label='Ord'); ax0.errorbar(scan_vals,imp_m,yerr=imp_e,fmt='s--',label='Imp')
        ax0.set_xlabel(var); ax0.set_ylabel('Avg hits'); ax0.set_ylim(bottom=0); ax0.grid(alpha=0.5); ax0.legend()
        fig.suptitle(f'Scan {var} ({steps} steps)')
        canvas.draw()

    btn_sim.config(command=run_sim); btn_scan.config(command=run_scan)
    root.mainloop()

if __name__=='__main__':
    build_gui()
