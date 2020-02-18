nohup ~/anaconda3/bin/python3.6 -u multi_classify.py >./model_save/log.txt 2>&1&
#nohup ~/anaconda3/bin/python3.6 -u handle.py >./log4.txt 2>&1&
tail -f model_save/log.txt
