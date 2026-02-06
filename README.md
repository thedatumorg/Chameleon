
<h2 id="start"> ✍️ Get Started </h2>

**Step 1:** Clone this repository using `git` and change into its root directory.

**Step 2:** Install the dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

<h2 id="start"> 🧑‍💻 Running Scripts </h2>

**Stage 1: Model Recommender** 

```bash
cd chameleon/MolRec

python train_U.py --domain [domain id] --save_dir [weight saving direc] or
python train_M.py --domain [domain id] --save_dir [weight saving direc]
```

**Stage 2: Model Optimizer**

```bash
cd benchmark_exp
python run_AutoAD_U_ranking.py --AutoAD_Name ChameleonOpt_precomputed --variant ID --save True --pretrained_weights [weight saving direc] --save_dir [eval result saving direc]
python run_AutoAD_U.py --AutoAD_Name ChameleonOpt_U_ID --variant [num of ens components] --save True --save_dir [eval result saving direc]
python run_AutoAD_U.py --AutoAD_Name ChameleonOpt_U_OOD --variant [num of ens components] --save True --save_dir [eval result saving direc]

python run_AutoAD_M_ranking.py --AutoAD_Name ChameleonOpt_precomputed --variant ID --save True --pretrained_weights [weight saving direc] --save_dir [eval result saving direc]
python run_AutoAD_M.py --AutoAD_Name ChameleonOpt_M_ID --variant [num of ens components] --save True --save_dir [eval result saving direc]
python run_AutoAD_M.py --AutoAD_Name ChameleonOpt_M_OOD --variant [num of ens components] --save True --save_dir [eval result saving direc]
```


**Evaluation**

```python
import pandas as pd
df = pd.read_csv(f'{[eval result saving direc]}/ChameleonOpt_U_ID_{[num of ens components]}.csv')
df = pd.read_csv(f'{[eval result saving direc]}/ChameleonOpt_U_OOD_{[num of ens components]}.csv')
df = pd.read_csv(f'{[eval result saving direc]}/ChameleonOpt_M_ID_{[num of ens components]}.csv')
df = pd.read_csv(f'{[eval result saving direc]}/ChameleonOpt_M_OOD_{[num of ens components]}.csv')
print(df.shape)
print(df['VUS-PR'].mean())
```

