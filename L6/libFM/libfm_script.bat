@ECHO OFF
perl triple_format_to_libfm.pl -in train.csv -target 3 -delete_column 0 -separator ","
perl triple_format_to_libfm.pl -in test.csv -target 3 -delete_column 0 -separator ","
libFM -task r -train train.csv.libfm -test test.csv.libfm -dim '1,1,8' -iter 1000 -method sgd -learn_rate 0.01 -regular '0,0,0.01' -init_stdev 0.1 -out movielens_sgd_out.txt
pause