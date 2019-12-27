dat=$(date +"%F_%H:%M:%S")
data='FDD'

python result_new.py -class Falls NotFalls -streams pose spatial temporal -fid ${data}-test -cid ${data}-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_PO.txt
python result_new.py -class Falls NotFalls -streams ritmo spatial temporal -fid ${data}-test -cid ${data}-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_RY.txt
python result_new.py -class Falls NotFalls -streams saliency spatial temporal -fid ${data}-test -cid ${data}-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_SA.txt

python result_new.py -class Falls NotFalls -streams spatial temporal -fid ${data}-test -cid ${data}-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB.txt

python result_new.py -class Falls NotFalls -streams pose temporal -fid ${data}-test -cid ${data}-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_PO.txt
python result_new.py -class Falls NotFalls -streams pose spatial -fid ${data}-test -cid ${data}-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RGB_PO.txt
python result_new.py -class Falls NotFalls -streams pose ritmo -fid ${data}-test -cid ${data}-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RY_PO.txt
python result_new.py -class Falls NotFalls -streams pose saliency -fid ${data}-test -cid ${data}-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_SA_PO.txt

python result_new.py -class Falls NotFalls -streams ritmo temporal -fid ${data}-test -cid ${data}-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RY.txt
python result_new.py -class Falls NotFalls -streams ritmo spatial -fid ${data}-test -cid ${data}-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RGB_RY.txt
python result_new.py -class Falls NotFalls -streams ritmo saliency -fid ${data}-test -cid ${data}-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_SA_RY.txt

python result_new.py -class Falls NotFalls -streams saliency temporal -fid ${data}-test -cid ${data}-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_SA.txt
python result_new.py -class Falls NotFalls -streams saliency spatial -fid ${data}-test -cid ${data}-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RGB_SA.txt

