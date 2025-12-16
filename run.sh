source venv/bin/activate

# mkdir -p logs submissions

# python3 precompute_splits.py

# python3 exp_mask_single.py >> logs/exp_mask_single.log
# python3 score.py submissions/exp_mask_single.csv >> logs/exp_mask_single.log

# python3 exp_mask.py >> logs/exp_mask.log
# python3 score.py submissions/exp_mask.csv >> logs/exp_mask.log

# python3 exp_mask_f1loss.py >> logs/exp_mask_f1loss.log
# python3 score.py submissions/exp_mask_f1loss.csv >> logs/exp_mask_f1loss.log

# python3 exp_mask_focal.py >> logs/exp_mask_focal.log
# python3 score.py submissions/exp_mask_focal.csv >> logs/exp_mask_focal.log

# python3 exp_embed.py >> logs/exp_embed.log
# python3 score.py submissions/exp_embed.csv >> logs/exp_embed.log

# python3 exp_split.py >> logs/exp_split.log
# python3 score.py submissions/exp_split.csv >> logs/exp_split.log

# python3 exp_contrastive_triplet.py >> logs/exp_contrastive_triplet.log
# python3 score.py submissions/exp_contrastive_triplet.csv >> logs/exp_contrastive_triplet.log

# python3 exp_contrastive_infonce.py >> logs/exp_contrastive_infonce.log
# python3 score.py submissions/exp_contrastive_infonce.csv >> logs/exp_contrastive_infonce.log

# python3 exp_contrastive_ntxent.py >> logs/exp_contrastive_ntxent.log
# python3 score.py submissions/exp_contrastive_ntxent.csv >> logs/exp_contrastive_ntxent.log

# python3 exp_mask_graph_based.py >> logs/exp_mask_graph_based.log
# python3 score.py submissions/exp_mask_graph_based.csv >> logs/exp_mask_graph_based.log