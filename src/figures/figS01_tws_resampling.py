from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter

root = Path(r'<PROJECT_ROOT>\Docs')
out = root / 'major_revision_analyses' / 'Figure_S3_full_precision_correlation_update'
source_csv = out / 'Figure_S3_full_precision_source_table.csv'
prev_rebuild = root / 'major_revision_analyses' / 'REBUILT_Figure_S3_TWS_resampling_sensitivity'
panel_b_path = prev_rebuild / 'tables' / 'figure_s3b_nrmse_table.csv'
panel_c_path = prev_rebuild / 'tables' / 'figure_s3c_resampling_sensitivity_table.csv'
land_order = ['Grasslands','Natural Forests','Savannas','Shrublands']
alt_order = ['Coarse fractional-overlap','5 km nearest overlay','Reverse block average','Stored masked coarse export']
altlab = ['Coarse\nfractional-\noverlap','5 km nearest\noverlay','Reverse block\naverage','Stored masked\ncoarse export']
source = pd.read_csv(source_csv)
panel_b = pd.read_csv(panel_b_path)
panel_c = pd.read_csv(panel_c_path)
a_vals = source.set_index('Land cover').loc[land_order, 'Pearson r full precision'].astype(float).to_numpy().reshape(4,1)
b_vals = panel_b.set_index('land_type').loc[land_order, 'nrmse'].to_numpy(float).reshape(4,1)
c_vals = panel_c.pivot(index='land_type', columns='alternative_representation', values='nrmse').loc[land_order, alt_order].to_numpy(float)

# Use a deliberately broad, high-correlation scale so tiny fifth/sixth-decimal differences
# do not create visually exaggerated contrast.
vmin, vmax = 0.99990, 1.00000
soft_green = LinearSegmentedColormap.from_list('soft_high_corr_green', ['#eef8ea', '#d6efd0', '#bce3b5', '#9bd59a'])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':16,'xtick.labelsize':12,'ytick.labelsize':13,'pdf.fonttype':42,'ps.fonttype':42})
fig = plt.figure(figsize=(18.8,8.15), facecolor='white')
fig.suptitle('Figure S3. Stability of TWS land-cover summaries across spatial representations', fontsize=22, fontweight='bold', y=.965)
fig.text(.5,.905,'Main analytical overlay: 5 km bilinear harmonization. Stability reference: coarse fractional-overlap land-cover summary.', ha='center', fontsize=15)
gs = fig.add_gridspec(1,8,left=.065,right=.965,bottom=.245,top=.82,width_ratios=[1.18,.052,.22,1.05,.045,.72,3.75,.065],wspace=.08)
axa=fig.add_subplot(gs[0,0]); cba=fig.add_subplot(gs[0,1]); axb=fig.add_subplot(gs[0,3]); cbb=fig.add_subplot(gs[0,4]); axc=fig.add_subplot(gs[0,6]); cbc=fig.add_subplot(gs[0,7])
def border(ax):
    for s in ax.spines.values():
        s.set_linewidth(.8); s.set_color('#222')
# Panel a: full precision values, soft color variation.
im=axa.imshow(a_vals,cmap=soft_green,norm=Normalize(vmin,vmax),aspect='auto')
axa.set_title('(a)',pad=8,fontsize=17)
axa.set_xticks([0],['Pearson r'],fontsize=13)
axa.set_yticks(range(4),['Grasslands','Natural\nForests','Savannas','Shrublands'],fontsize=14)
axa.tick_params(length=0)
for i,v in enumerate(a_vals[:,0]):
    axa.text(0,i,f'{v:.8f}',ha='center',va='center',fontsize=13.2,fontweight='bold',color='black')
cb=fig.colorbar(im,cax=cba)
cb.ax.tick_params(labelsize=10)
cb.set_label('Pearson r\n(soft high-r scale)',fontsize=11,labelpad=9)
cb.formatter=FormatStrFormatter('%.5f')
cb.set_ticks([0.99990,0.999925,0.99995,0.999975,1.00000])
cb.update_ticks()
border(axa)
# Panel b unchanged.
im=axb.imshow(b_vals,cmap='YlOrRd',norm=Normalize(0,.030),aspect='auto')
axb.set_title('(b)',pad=8,fontsize=17)
axb.set_xticks([0],['nRMSE'],fontsize=13)
axb.set_yticks(range(4),['','','',''])
axb.tick_params(length=0)
for i,v in enumerate(b_vals[:,0]):
    axb.text(0,i,f'{v:.3f}',ha='center',va='center',fontsize=15,fontweight='bold')
cb=fig.colorbar(im,cax=cbb)
cb.ax.tick_params(labelsize=11); cb.set_ticks([0,.005,.010,.015,.020,.025,.030])
border(axb)
# Panel c unchanged.
im=axc.imshow(c_vals,cmap='YlOrRd',norm=Normalize(0,.55),aspect='auto')
axc.set_title('(c)',pad=8,fontsize=17)
axc.set_xticks(range(4),altlab,fontsize=13)
axc.set_yticks(range(4),['Grasslands','Natural\nForests','Savannas','Shrublands'],fontsize=14)
axc.tick_params(length=0,pad=8)
for i in range(4):
    for j in range(4):
        axc.text(j,i,f'{c_vals[i,j]:.3f}',ha='center',va='center',fontsize=14,fontweight='bold',color=('white' if c_vals[i,j]>=.35 else 'black'))
cb=fig.colorbar(im,cax=cbc)
cb.ax.tick_params(labelsize=11); cb.set_label('nRMSE relative to main 5 km bilinear overlay',fontsize=13,labelpad=12); cb.set_ticks([0,.1,.2,.3,.4,.5])
border(axc)
fig.text(.5,.085,'Higher Pearson r and lower nRMSE indicate stronger agreement. These checks assess land-cover monthly summaries only, not independent fine-scale TWS information.',ha='center',fontsize=13)
png = out / 'Figure_S3_full_precision_soft_palette.png'
pdf = out / 'Figure_S3_full_precision_soft_palette.pdf'
fig.savefig(png, bbox_inches='tight', dpi=300, facecolor='white')
fig.savefig(pdf, bbox_inches='tight', facecolor='white')
plt.close(fig)
note = out / 'Figure_S3_soft_palette_note.md'
note.write_text(
    '# Figure S3 Soft Palette Note\n\n'
    'Generated: 2026-07-27\n\n'
    'This is a formatting-only update to the full-precision Figure S3. No data, Pearson r values, nRMSE values, source tables, method labels, comparison logic, or manuscript/supplementary files were changed.\n\n'
    'Panel (a) was redrawn with a broader high-correlation scale (0.99990 to 1.00000) and a soft green palette so the extremely close Pearson r values do not appear visually over-separated. Panels (b) and (c) were preserved from the same source values.\n\n'
    f'- PNG: `{png}`\n'
    f'- PDF: `{pdf}`\n',
    encoding='utf-8'
)
print(png)
print(pdf)
print(note)
