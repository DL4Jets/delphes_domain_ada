import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pdb import set_trace

parser = ArgumentParser()
parser.add_argument('inputdir')
parser.add_argument("-p",  dest='postfix', default='', help="plot postfix")
parser.add_argument("-c",  dest='compare', default='stepwise_domain_adaptation', help="training to compare data and MC to")
args = parser.parse_args()

ntraingings = 5

#
# Losses
#
# pd.DataFrame(np.load(  '../domada_50_epochs_newsample/domain_adaptation_two_samples/history.npy'))
da_history   = pd.DataFrame(np.load('%s/%s/history.npy' % (args.inputdir,args.compare)))
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

databtag='btag_discriminator_loss_2'
#'btag_discriminator_'

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
	
	
	nepochs=da_history['val_btag_discriminator_'+idstring+'_2_mean'].shape[0]
	plt.plot(da_history['val_btag_discriminator_'+idstring+'_2_mean'],label=dataonDA, c='blue')
	if fill:
		plt.fill_between(
			range(nepochs), 
			da_history['val_btag_discriminator_'+idstring+'_2_mean']-da_history['val_btag_discriminator_'+idstring+'_2_std'], 
			da_history['val_btag_discriminator_'+idstring+'_2_mean']+da_history['val_btag_discriminator_'+idstring+'_2_std'], 
			color='blue',
			alpha=0.3
			)
	plt.plot(da_history['val_btag_discriminator_'+idstring+'_1_mean'],label=mconDA, c='green',linestyle=':')
	if fill:
		plt.fill_between(
			range(nepochs), 
			da_history['val_btag_discriminator_'+idstring+'_1_mean']-da_history['val_btag_discriminator_'+idstring+'_1_std'], 
			da_history['val_btag_discriminator_'+idstring+'_1_mean']+da_history['val_btag_discriminator_'+idstring+'_1_std'], 
			color='green',
			alpha=0.3
			)
	
	plt.plot(mc_history['val_btag_discriminator_'+idstring+'_2_mean'],label=dataonmc, c='red')
	if fill:
		plt.fill_between(
			range(nepochs), 
			mc_history['val_btag_discriminator_'+idstring+'_2_mean']-mc_history['val_btag_discriminator_'+idstring+'_2_std'], 
			mc_history['val_btag_discriminator_'+idstring+'_2_mean']+mc_history['val_btag_discriminator_'+idstring+'_2_std'], 
			color='red',
			alpha=0.3
			)
	plt.plot(mc_history['val_btag_discriminator_'+idstring+'_1_mean'],label=mconmc, c='blueviolet',linestyle=':')
	if fill:
		plt.fill_between(
			range(nepochs), 
			mc_history['val_btag_discriminator_'+idstring+'_1_mean']-mc_history['val_btag_discriminator_'+idstring+'_1_std'], 
			mc_history['val_btag_discriminator_'+idstring+'_1_mean']+mc_history['val_btag_discriminator_'+idstring+'_1_std'], 
			color='blueviolet',
			alpha=0.3
			)
	plt.plot(data_history['val_btag_discriminator_'+idstring+'_2_mean'],label=dataondata, c='orange')
	plt.plot(data_history['val_btag_discriminator_'+idstring+'_1_mean'],label=mcondata, c='brown',linestyle=':')
	
	if idstring == 'weighted_acc':
		plt.plot(da_history['datamc_discriminator_'+idstring+'_1_mean'],label='data/mc discr', c='fuchsia',linestyle='--')
	
	plt.ylabel(''+idstring+'')
	plt.xlabel('epochs')
	plt.legend(ncol=2, loc=1)#'best')
	textonly(plt.gca(),metaleg,loc=3)
	fig.savefig('%s/%s%s.png' % (args.inputdir, idstring, args.postfix))
	fig.savefig('%s/%s%s.pdf' % (args.inputdir, idstring, args.postfix))
	plt.clf()
	

makeEpochPlot('loss',True)
makeEpochPlot('weighted_binary_accuracy',False)

#plot weights
plt.clf()
plt.plot(da_history.index, da_history.weight_mean, label='fitted weight', color='blue')
plt.fill_between(
	da_history.index,
	da_history.weight_mean - da_history.weight_std,
	da_history.weight_mean + da_history.weight_std,
	color='blue', alpha=0.3
	)
plt.plot([da_history.index.min(), da_history.index.max()], [da_history.real_weight_mean, da_history.real_weight_mean], label='best value', ls='--')
plt.xlabel('epoch')
plt.ylabel('weight')
#plt.legend(loc='best')
plt.grid(True)
plt.savefig('%s/weights%s.png' % (args.inputdir, args.postfix))

from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import InterpolatedUnivariateSpline
from pdb import set_trace

## pd.DataFrame(np.load(  '../domada_50_epochs_newsample/domain_adaptation_two_samples/predictions.npy'))
da_predictions   = pd.DataFrame(np.load('%s/%s/predictions.npy' % (args.inputdir, args.compare)))
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
