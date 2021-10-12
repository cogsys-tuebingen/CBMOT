python main.py --work_dir work_dir1 --max_age 3 --Lidar_traindata True
python centerTrack_id_rearrangement.py --work_dir work_dir2 --checkpoint resources/centertrack_origin.json
python main.py --work_dir work_dir3  --score_decay 0.2 --score_update multiplication
python main.py --work_dir work_dir4  --checkpoint2 resources/centertrack_tracks.json --score_decay 0.2 --score_update multiplication --fusion True --star True
python main.py --work_dir work_dir5  --checkpoint2 resources/centertrack_tracks.json --score_decay 0.0 --score_update nn --model_path resources/hl4_LeakyReLU_hu100_e1250_2i.pth --max_age 6 --fusion True --star True
