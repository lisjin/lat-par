dataset=../data/AMR/amr_2.0
SUF="_ldc_f"
python3 work.py --test_data ${dataset}/test.txt.features.preproc.json\
                      --test_forests $SCR/k-decomp/ldc/test_forests${SUF}.hdf5\
                      --test_sep2frags $SCR/k-decomp/ldc/test_sep2frags${SUF}.pkl\
               --test_batch_size 44444\
               --load_path $1\
               --beam_size 10\
               --alpha 1.6\
               --max_time_step 100\
               --output_suffix _test_out
