import sys
import os
from pathlib import Path
def args_to_dict(argv):
    arg_dict = {}
    for i, name in enumerate(argv[:-1]):
        if name.startswith("--") and not argv[i+1].startswith("--"):
            arg_dict[name[2:]] = argv[i+1]
    return arg_dict

def dict_to_args(arg_dict):
    return " ".join([f"--{name} {value}" if " " not in str(value) else f"--{name} \"{value}\"" for name, value in arg_dict.items()])

count = sys.argv[1]
name = sys.argv[2]
file = sys.argv[3]
argv = sys.argv[4:]

arg_dict = args_to_dict(argv)
# add the current path + logs to the output path
if "output" in arg_dict:
    arg_dict["output"] = Path(os.getcwd()) / "logs" / arg_dict["output"]
    Path(arg_dict["output"]).mkdir(parents=True, exist_ok=True)

for key in sorted(list(arg_dict.keys()), key=len, reverse=True):
    name = name.replace("@"+key, str(arg_dict[key]))
command = f"python {file} {dict_to_args(arg_dict)}"

with open("command_history.txt", "a") as fp:
    fp.write(" ".join(sys.argv)+"\n")

file_content = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account=rrg-afyshe
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=64000M
#SBATCH --output={arg_dict["output"]}/%x-%j.txt
#SBATCH --array=1-{count}

echo "load"
module load python/3.6 cuda cudnn
echo "create env"
virtualenv --no-download  $SLURM_TMPDIR/tensorflow_env
source $SLURM_TMPDIR/tensorflow_env
pip install --no-index --upgrade pip
pip install --no-index tensorflow_gpu numpy pandas matplotlib pyyaml tensorflow_datasets
echo "clone"
git clone . $SLURM_TMPDIR/repo
echo "copy"
cd $SLURM_TMPDIR/repo
echo "run"
pwd
ls
echo "run $(date '+%d/%m/%Y %H:%M:%S')"
{command}
echo "done"
"""

print(file_content)

with open("job.sh", "w") as fp:
    fp.write(file_content)

os.system("sbatch job.sh")
