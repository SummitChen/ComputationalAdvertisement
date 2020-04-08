@ECHO OFF
java -jar bin/RankLib-patched.jar -train MQ2008/Fold1/train.txt -test MQ2008/Fold1/test.txt -validate MQ2008/Fold1/vali.txt -ranker 2 -metric2t NDCG@10 -metric2T NDCG@10 -save models/mq_ranknet.txt
java -jar bin/RankLib-patched.jar -train MQ2008/Fold1/train.txt -test MQ2008/Fold1/test.txt -validate MQ2008/Fold1/vali.txt -ranker 6 -metric2t NDCG@10 -metric2T NDCG@10 -save models/mq_lambdamart.txt
java -jar bin/RankLib-patched.jar -train MQ2008/Fold1/train.txt -test MQ2008/Fold1/test.txt -validate MQ2008/Fold1/vali.txt -ranker 8 -metric2t NDCG@10 -metric2T NDCG@10 -save models/mq_listnet.txt

java -jar bin/RankLib-patched.jar -test MQ2008/Fold1/test.txt -metric2T NDCG@10 -idv output/baseline.ndcg.txt
java -jar bin/Ranklib-patched.jar -load models/mq_ranknet.txt -test MQ2008/Fold1/test.txt -metric2T NDCG@10 -idv output/rn.ndcg.txt
java -jar bin/Ranklib-patched.jar -load models/mq_lambdamart.txt -test MQ2008/Fold1/test.txt -metric2T NDCG@10 -idv output/lm.ndcg.txt
java -jar bin/Ranklib-patched.jar -load models/mq_listnet.txt -test MQ2008/Fold1/test.txt -metric2T NDCG@10 -idv output/ln.ndcg.txt

java -cp bin/RankLib-patched.jar ciir.umass.edu.eval.Analyzer -all output/ -base baseline.ndcg.txt > analysis.txt
pause