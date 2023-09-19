import pandas as pd
import wandb

WANDB_API_KEY = "a1ffdc7f846152c3339870a8ab234dedca6b75c6"

api = wandb.Api()
entity, project = "smallstarting", "scribblekitti_mix_cyh"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    print(run.name)
    if not run.name.startswith("cylinder3d_mt_mix_unc_"):
        continue
    # .summary contains output keys/values for
    # metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)
    
writer=pd.ExcelWriter('project.xlsx', engine='openpyxl')
    
output = {}

for i in range(len(summary_list)):
    for key in summary_list[i].keys():
        if key.startswith("val_teacher") or key.startswith("val_best"):
            if not key in output.keys():
                output[key] = []
            output[key].append(summary_list[i][key])
output["name"] = name_list
runs_df = pd.DataFrame(output)

runs_df.to_excel(writer, sheet_name="teacher", float_format="%.4f")

output = {}

for i in range(len(summary_list)):
    for key in summary_list[i].keys():
        if key.startswith("val_student") or key.startswith("val_best"):
            if not key in output.keys():
                output[key] = []
            output[key].append(summary_list[i][key])
output["name"] = name_list
runs_df = pd.DataFrame(output)

runs_df.to_excel(writer, sheet_name="student", float_format="%.4f")

output = {}

for i in range(len(summary_list)):
    for key in summary_list[i].keys():
        if key.startswith("ls_loss") or key.startswith("cl_loss") or key.startswith("val_loss") or key.startswith("mix_loss"):
            if not key in output.keys():
                output[key] = []
            output[key].append(summary_list[i][key])
output["name"] = name_list
runs_df = pd.DataFrame(output)

runs_df.to_excel(writer, sheet_name="loss", float_format="%.4f")

# writer.save()
writer.close()