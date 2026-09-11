import json

with open('train_notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 5: Data compilation
cell5 = nb['cells'][4]
cell5['source'] = [
    "# 5. [GIAI ĐOẠN 1: TỔNG HỢP MASTER DATASET 45K (COMMUNITY + SYNTHETIC ANCHOR)]\n",
    "import os\n",
    "print('💎 [1/3] Biên dịch Golden Dense Core...')\n",
    "!python data_pipeline/build_dense_reasoning_core.py dataset/dense_core_vi.jsonl\n",
    "print('💬 [2/3] Biên dịch Multi-turn Dialogue Core...')\n",
    "!python data_pipeline/generate_multiturn_data.py\n",
    "print('📚 [3/3] Tải và tích hợp cộng đồng (GSM8K, UIT-ViQuAD, Multi-turn Alpaca)...')\n",
    "!python data_pipeline/download_community_datasets.py\n",
    "\n",
    "master_file = 'dataset/vimind_4.5_master.jsonl'\n",
    "if not os.path.exists(master_file):\n",
    "    master_file = 'dataset/vimind_4.5_combined.jsonl'\n",
    "!ls -lh dataset/*.jsonl\n"
]

# Cell 7: Native SFT
cell7 = nb['cells'][6]
cell7['source'] = [
    "# 7. [GIAI ĐOẠN 2: HUẤN LUYỆN SFT NATIVE (Master Dataset 45k)]\n",
    "import os\n",
    "data_path = 'dataset/vimind_4.5_master.jsonl' if os.path.exists('dataset/vimind_4.5_master.jsonl') else 'dataset/dense_core_vi.jsonl'\n",
    "base_checkpoint = 'out/pretrain/vimind_4.0_base_final' if os.path.exists('out/pretrain/vimind_4.0_base_final') else ('out/pretrain/vimind_4.0_base.pth' if os.path.exists('out/pretrain/vimind_4.0_base.pth') else 'none')\n",
    "!python -u trainer/train_sft.py \\\n",
    "    --data_path {data_path} \\\n",
    "    --from_pretrained {base_checkpoint} \\\n",
    "    --tokenizer_dir model \\\n",
    "    --save_dir out/sft_moe \\\n",
    "    --save_weight vimind_4.0_moe \\\n",
    "    --use_moe \\\n",
    "    --num_experts 4 \\\n",
    "    --num_experts_per_tok 2 \\\n",
    "    --batch_size 8 \\\n",
    "    --accumulation_steps 8 \\\n",
    "    --max_seq_len 512 \\\n",
    "    --gradient_checkpointing \\\n",
    "    --epochs 3 \\\n",
    "    --max_steps 3000 \\\n",
    "    --learning_rate 2e-4 \\\n",
    "    --dtype float16 \\\n",
    "    --log_interval 50 \\\n",
    "    --save_interval 1000\n",
    "\n",
    "!rm -rf out/sft_moe/*_step_*\n",
    "!ls -lh out/sft_moe\n"
]

# Cell 8: Pro SFT
cell8 = nb['cells'][7]
cell8['source'] = [
    "# 8. [TRACK 2: HUẤN LUYỆN VIMIND 4.5 PRO (0.5B Foundation Model Alignment)]\n",
    "import os\n",
    "data_path = 'dataset/vimind_4.5_master.jsonl' if os.path.exists('dataset/vimind_4.5_master.jsonl') else 'dataset/dense_core_vi.jsonl'\n",
    "!python -u trainer/train_pro_sft.py \\\n",
    "    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \\\n",
    "    --data_path {data_path} \\\n",
    "    --output_dir /kaggle/working/vimind_4.0_pro_final \\\n",
    "    --epochs 2 \\\n",
    "    --max_steps 1000 \\\n",
    "    --batch_size 4 \\\n",
    "    --accumulation_steps 8 \\\n",
    "    --learning_rate 2e-5 \\\n",
    "    --fp16\n",
    "\n",
    "!ls -lh /kaggle/working/vimind_4.0_pro_final\n"
]

with open('train_notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('SUCCESS UPDATING train_notebook.ipynb')
