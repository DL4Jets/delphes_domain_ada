import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('inputdir')
parser.add_argument("-p",  dest='postfix', default='', help="plot postfix")
args = parser.parse_args()

ntraingings=5

#
# Losses
#
#pd.DataFrame(np.load(  '../domada_50_epochs_newsample/domain_adaptation_two_samples/history.npy'))
da_history   = pd.DataFrame(np.load('%s/stepwise_domain_adaptation/history.npy' % args.inputdir))
data_history = pd.DataFrame(np.load('%s/data_training/history.npy' % args.inputdir))
mc_history   = pd.DataFrame(np.load('%s/MC_training/history.npy' % args.inputdir))

fig = plt.figure()

dataonDA='$\\bf{data}$ on $\it{D.A.}$'
mconDA='$\\bf{mc}$ on $\it{D.A.}$'
dataonmc='$\\bf{data}$ on $\it{mc}$'
mconmc='$\\bf{mc}$ on $\it{mc}$'
dataondata='$\\bf{data}$ on $\it{data}$'
mcondata='$\\bf{mc}$ on $\it{data}$'

metaleg='$\\bf{sample}$\n$\it{training}$'



from mpl_toolkits.axes_grid.anchored_artists import AnchoredText


def textonly(ax, txt, fontsize = 10, loc = 2, *args, **kwargs):
	at = AnchoredText(txt,
	                  prop=dict(size=fontsize), 
	                  frameon=True,
	                  loc=loc)
	at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
	ax.add_artist(at)
	return at

def makeEpochPlot(idstring,fill):
	
	
	if idstring == 'weighted_acc':
		plt.ylim(0.4, 1.1)
	else:
		plt.ylim(0.2, 0.45)
	
	
	nepochs=da_history['val_data_'+idstring+'_mean'].shape[0]
	plt.plot(da_history['val_data_'+idstring+'_mean'],label=dataonDA, c='blue')
	if fill:
		plt.fill_between(
			range(nepochs), 
			da_history['val_data_'+idstring+'_mean']-da_history['val_data_'+idstring+'_std'], 
			da_history['val_data_'+idstring+'_mean']+da_history['val_data_'+idstring+'_std'], 
			color='blue',
			alpha=0.3
			)
	plt.plot(da_history['val_mc_'+idstring+'_mean'],label=mconDA, c='green',linestyle=':')
	if fill:
		plt.fill_between(
			range(nepochs), 
			da_history['val_mc_'+idstring+'_mean']-da_history['val_mc_'+idstring+'_std'], 
			da_history['val_mc_'+idstring+'_mean']+da_history['val_mc_'+idstring+'_std'], 
			color='green',
			alpha=0.3
			)
	
	plt.plot(mc_history['val_data_'+idstring+'_mean'],label=dataonmc, c='red')
	plt.plot(mc_history['val_mc_'+idstring+'_mean'],label=mconmc, c='blueviolet',linestyle=':')
	plt.plot(data_history['val_data_'+idstring+'_mean'],label=dataondata, c='orange')
	plt.plot(data_history['val_mc_'+idstring+'_mean'],label=mcondata, c='brown',linestyle=':')
	
	if idstring == 'weighted_acc':
		plt.plot(da_history['val_Add_'+idstring+'_mean'],label='data/mc discr', c='fuchsia',linestyle='--')
	
	plt.ylabel(''+idstring+'')
	plt.xlabel('epochs')
	plt.legend(ncol=2, loc=1)#'best')
	textonly(plt.gca(),metaleg,loc=3)
	fig.savefig('%s/%s%s.png' % (args.inputdir, idstring, args.postfix))
	fig.savefig('%s/%s%s.pdf' % (args.inputdir, idstring, args.postfix))
	plt.clf()
	

makeEpochPlot('loss',True)
makeEpochPlot('weighted_acc',False)



## da_w50_l25_hist = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w50_l.25/history.npy'))
## plt.plot(da_w50_l25_hist['val_data_loss_mean'],label='data DA w50 l0.25', c='blue' , ls='--')
## plt.plot(da_w50_l25_hist['val_mc_loss_mean'  ],label='mc DA w50 l0.25'  , c='green', ls='--')
## 
## da_w50_l04_hist = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w50_l.04/history.npy'))
## plt.plot(da_w50_l04_hist['val_data_loss_mean'],label='data DA w50 l0.04', c='blue' , ls='-.')
## plt.plot(da_w50_l04_hist['val_mc_loss_mean'  ],label='mc DA w50 l0.04'  , c='green', ls='-.')
## 
## da_w25_l50_hist = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w25_l.5/history.npy'))
## plt.plot(da_w25_l50_hist['val_data_loss_mean'],label='data DA w25 l0.5' , c='blue' , ls=':')
## plt.plot(da_w25_l50_hist['val_mc_loss_mean'  ],label='mc DA w25 l0.5'   , c='green', ls=':')
## 
## da_w05_l01_hist = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w05_l1/history.npy'))
## plt.plot(da_w05_l01_hist['val_data_loss_mean'],label='data DA w5 l1'    , c='cyan' , ls='-')
## plt.plot(da_w05_l01_hist['val_mc_loss_mean'  ],label='mc DA w5 l1'      , c='limegreen', ls='-')



#
# ROCs
#


from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import InterpolatedUnivariateSpline
from pdb import set_trace

## pd.DataFrame(np.load(  '../domada_50_epochs_newsample/domain_adaptation_two_samples/predictions.npy'))
da_predictions   = pd.DataFrame(np.load('%s/stepwise_domain_adaptation/predictions.npy' % args.inputdir))
data_predictions = pd.DataFrame(np.load('%s/data_training/predictions.npy' % args.inputdir))
mc_predictions   = pd.DataFrame(np.load('%s/MC_training/predictions.npy' % args.inputdir))

def draw_roc(df, label, color, draw_unc=False, ls='-', draw_auc=True):
	newx = np.logspace(-3, 0, 50)#arange(0,1,0.01)
	
	tprs = pd.DataFrame()
	scores = []
	for idx in range(ntraingings):
		tmp_fpr, tmp_tpr, _ = roc_curve(df.isB, df['prediction_%d' % idx])
		scores.append(
			roc_auc_score(df.isB, df['prediction_%d' % idx])
			)
		coords = pd.DataFrame()
		coords['fpr'] = tmp_fpr
		coords['tpr'] = tmp_tpr
		clean = coords.drop_duplicates(subset=['fpr'])
		spline = InterpolatedUnivariateSpline(clean.fpr, clean.tpr,k=1)
		tprs[idx] = spline(newx)
	scores = np.array(scores)
	auc = ' AUC: %.3f +/- %.3f' % (scores.mean(), scores.std()) if draw_auc else ''
	if draw_unc:
		plt.fill_between(
			newx,
			tprs.mean(axis=1) - tprs.std(axis=1),
			tprs.mean(axis=1) + tprs.std(axis=1),
			color=color,
			alpha=0.3
			)		
	
	plt.plot(newx, tprs.mean(axis=1), label=label + auc, c=color, ls=ls)
	
	
plt.clf()
draw_roc(
	da_predictions[da_predictions.isMC == 0],
	dataonDA,
	'blue',
	draw_unc = True,
	draw_auc=True,
	)
draw_roc(
	da_predictions[da_predictions.isMC == 1],
	mconDA,
	'green',
	draw_unc = True,
	draw_auc=True,
	ls=':'
	)

## da_w50_l25_pred = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w50_l.25/predictions.npy'))
## draw_roc(da_w50_l25_pred[da_w50_l25_pred.isMC == 0], 'data DA w50 l0.25', 'blue', ls='--', draw_auc=True)
## draw_roc(da_w50_l25_pred[da_w50_l25_pred.isMC == 1], 'mc DA w50 l0.25'  , 'green', ls='--', draw_auc=True)
## 
## da_w50_l04_pred = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w50_l.04/predictions.npy'))
## draw_roc(da_w50_l04_pred[da_w50_l04_pred.isMC == 0], 'data DA w50 l0.04', 'blue' , ls='-.', draw_auc=True)
## draw_roc(da_w50_l04_pred[da_w50_l04_pred.isMC == 1], 'mc DA w50 l0.04'  , 'green', ls='-.', draw_auc=True)
## 
## da_w25_l50_pred = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w25_l.5/predictions.npy'))
## draw_roc(da_w25_l50_pred[da_w25_l50_pred.isMC == 0], 'data DA w25 l0.5', 'blue' , ls=':', draw_auc=True)
## draw_roc(da_w25_l50_pred[da_w25_l50_pred.isMC == 1], 'mc DA w25 l0.5'  , 'green', ls=':', draw_auc=True)
## 
## da_w05_l01_pred = pd.DataFrame(np.load('../domada_50_epochs_newsample/domain_adaptation_two_samples_w05_l1/predictions.npy'))
## draw_roc(da_w05_l01_pred[da_w05_l01_pred.isMC == 0], 'data DA w5 l1', 'cyan', ls='-', draw_auc=True)
## draw_roc(da_w05_l01_pred[da_w05_l01_pred.isMC == 1], 'mc DA w5 l1'  , 'limegreen', ls='-', draw_auc=True)

draw_roc(
	mc_predictions[mc_predictions.isMC == 0],
	dataonmc, 'red', draw_auc=True
	)
draw_roc(
	mc_predictions[mc_predictions.isMC == 1],
	mconmc, 'blueviolet', draw_auc=True, ls=':'
	)

draw_roc(
	data_predictions[data_predictions.isMC == 0],
	dataondata, 'orange', draw_auc=True
	)
draw_roc(
	data_predictions[data_predictions.isMC == 1],
	mcondata, 'brown', draw_auc=True, ls=':'
	)

plt.xlim(0., 1)
plt.ylim(0.45, 1)
plt.grid(True)
plt.ylabel('true positive rate')
plt.xlabel('false positive rate')
plt.legend(loc='best')
textonly(plt.gca(),metaleg,loc=3)
fig.savefig('%s/rocs%s.png' % (args.inputdir, args.postfix))
fig.savefig('%s/rocs%s.pdf' % (args.inputdir, args.postfix))

plt.xlim(10**-3, 1)
plt.ylim(0.3, 1)
plt.gca().set_xscale('log')
fig.savefig('%s/rocs_log%s.png' % (args.inputdir, args.postfix))
fig.savefig('%s/rocs_log%s.pdf' % (args.inputdir, args.postfix))


def plot_discriminator(df, name):
	plt.clf()
	plt.hist(
		[df[df.isB == 1].prediction_mean, df[df.isB == 0].prediction_mean],
		bins = 50, range=(0, 1.), histtype='bar', stacked=True,
		color=['green', 'blue'], label=['B jets', 'light jets']
		)
	plt.ylabel('occurrences')
	plt.xlabel('NN output (averaged)')
	plt.legend(loc='best')
	fig.savefig('%s/%s%s.png' % (args.inputdir, name, args.postfix))
	fig.savefig('%s/%s%s.pdf' % (args.inputdir, name, args.postfix))

plot_discriminator(da_predictions[da_predictions.isMC == 1], 'nn_out_da_mc')
plot_discriminator(da_predictions[da_predictions.isMC == 0], 'nn_out_da_data')

plot_discriminator(data_predictions[data_predictions.isMC == 1], 'nn_out_dataTraining_mc')
plot_discriminator(data_predictions[data_predictions.isMC == 0], 'nn_out_dataTraining_data')

plot_discriminator(mc_predictions[mc_predictions.isMC == 1], 'nn_out_mcTraining_mc')
plot_discriminator(mc_predictions[mc_predictions.isMC == 0], 'nn_out_mcTraining_data')
