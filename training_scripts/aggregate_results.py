"""
This script serves as a way to create csv files with the results of
the training. These csv files can then be used with pandas
"""
import locgame.save_io as locio
import ml_utils.save_io as mlio
import ml_utils.analysis as mlanl
from ml_utils.utils import try_key
import pandas as pd
import os
import sys

if __name__ == "__main__":
    argued_folders = sys.argv[1:]
    model_folders = []
    for folder in argued_folders:
        if not mlio.is_model_folder(folder):
            model_folders += mlio.get_model_folders(folder,True)
        else:
            model_folders += [folder]
    print("Model Folders:", model_folders)
    for model_folder in model_folders:
        checkpts = mlio.get_checkpoints(model_folder)
        if len(checkpts) == 0: continue
        table = mlanl.get_table(mlio.load_checkpoint(checkpts[0]))
        for checkpt_path in checkpts:
            try:
                checkpt = mlio.load_checkpoint(checkpt_path)
                for k in table.keys():
                    if k in set(checkpt.keys()):
                        table[k].append(checkpt[k])
            except Exception as e:
                print("model folder:", model_folder)
                print("chckpt:", checkpt_path)

        df = pd.DataFrame(table)
        # These keys are ignored
        ignores = {"del_prev_sd","key_descriptions",
                   "search_keys","float_params"}
        for k,v in checkpt['hyps'].items():
            # Note that any key ending with an underscore is ignored
            if k not in ignores and k[-1] != "_":
                try:
                    df[k] = v
                except:
                    df[k] = str(v)
        params = checkpt['hyps']['float_params']
        for k,v in params.items():
            df[k] = v

        df['model_type'] = "Unk"
        idx = (df["egoCentered"]>=1)&(df["absoluteCoords"]>=1)
        df.loc[idx,"model_type"] = "EgoAbsolute"
        idx = (df["egoCentered"]>=1)&(df["absoluteCoords"]<=0)
        df.loc[idx,"model_type"] = "EgoRelative"
        idx = (df["egoCentered"]<=0)&(df["absoluteCoords"]>=1)
        df.loc[idx,"model_type"] = "AlloAbsolute"
        idx = (df["egoCentered"]<=0)&(df["absoluteCoords"]<=0)
        df.loc[idx,"model_type"] = "AlloRelative"
        save_path = os.path.join(model_folder, "model_data.csv")
        df.to_csv(save_path, sep="!", index=False, header=True)
