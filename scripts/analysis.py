import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from miseval import evaluate

from scripts.train import load_model, predict_3D


def save_results(model_name, device, metadata, vendor_datasets_3D, show_example=False):
  model = load_model(model_name)[0]

  results = metadata[['Vendor', 'Centre']].copy()

  results = results.assign(Dice_BG_ED=None, Dice_LV_ED=None, Dice_MYO_ED=None, Dice_RV_ED=None)
  results = results.assign(Dice_BG_ES=None, Dice_LV_ES=None, Dice_MYO_ES=None, Dice_RV_ES=None)

  results = results.assign(IoU_BG_ED=None, IoU_LV_ED=None, IoU_MYO_ED=None, IoU_RV_ED=None)
  results = results.assign(IoU_BG_ES=None, IoU_LV_ES=None, IoU_MYO_ES=None, IoU_RV_ES=None)

  # results = results.assign(HD_BG_ED=None, HD_LV_ED=None, HD_MYO_ED=None, HD_RV_ED=None)
  # results = results.assign(HD_BG_ES=None, HD_LV_ES=None, HD_MYO_ES=None, HD_RV_ES=None)


  for dataset_3D in vendor_datasets_3D:
      for i in range(len(dataset_3D)):
          # Get data for one subject
          subject = dataset_3D[i]
          labels = subject.seg.data 
          id = subject.id
          vendor = metadata.loc[id].Vendor
          
          # Make predictions
          labels_stacked = labels.permute((0,3,1,2)).long()
          ed_labels = labels_stacked[0]
          es_labels = labels_stacked[1]


          predictions = predict_3D(model, subject, device)

          ed_predictions = predictions[:predictions.shape[0]//2]
          es_predictions = predictions[predictions.shape[0]//2:]

          # Run multi-class evaluation
          dc_ed = list(evaluate(ed_labels, ed_predictions, metric="Dice", multi_class=True, n_classes=4))
          dc_es = list(evaluate(es_labels, es_predictions, metric="Dice", multi_class=True, n_classes=4))

          jc_ed = list(evaluate(ed_labels, ed_predictions, metric="Jaccard", multi_class=True, n_classes=4))
          jc_es = list(evaluate(es_labels, es_predictions, metric="Jaccard", multi_class=True, n_classes=4))

          # hd_ed = list(evaluate(ed_labels, ed_predictions, metric="AHD", multi_class=T07/10/1964   
          results.loc[id, results.columns[2:]]  = dc_ed + dc_es + jc_ed + jc_es


  results.to_csv(f'Results/{model_name}.csv', index=True)

  if show_example:
    fig, axes = plt.subplots(4, 12)
    axes[0, 5].set_title("ED Ground Truth")
    axes[1, 5].set_title("ED Predictions")
    axes[2, 5].set_title("ES Ground Truth")
    axes[3, 5].set_title("ES Predictions")

    for i in range(12):
        axes[0, i].imshow(ed_labels[i])
        axes[1, i].imshow(ed_predictions[i])
        axes[2, i].imshow(es_labels[i])
        axes[3, i].imshow(es_predictions[i])



def plot_metric_results(results, metric, ax, title):
    sns.boxplot(data = results.melt(id_vars = 'Centre',
                  value_vars = [f'{metric}_LV_ED',
                                f'{metric}_LV_ES',
                                f'{metric}_MYO_ED',
                                f'{metric}_MYO_ES',
                                f'{metric}_RV_ED',              
                                f'{metric}_RV_ES'],
                  var_name = 'Region'),
                hue = 'Centre',
                x = 'Region',
                y = 'value',
                ax=ax)
    ax.set_title(title)
    ax.grid(True)
    return ax



def show_results(model_name):
    results = pd.read_csv(f'Results/{model_name}.csv', index_col=0)
    grouped_by_vendor = results.groupby(['Vendor', 'Centre']).mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1 = plot_metric_results(results, "Dice", ax1, "Dice Coefficient")
    ax2 = plot_metric_results(results, "IoU", ax2, "IoU")

    plt.subplots_adjust(wspace=0.3)
    plt.show()
    return grouped_by_vendor



def compare_results():
   return 0



def main():
  return 0


if __name__ == '__main__':
  main()