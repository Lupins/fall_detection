dat=$(date +"%F_%H:%M:%S")
data='URFD'

python result_new.py -class Falls NotFalls -streams pose spatial temporal -fid URFD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_PO.txt
python result_new.py -class Falls NotFalls -streams ritmo spatial temporal -fid URFD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_RY.txt
python result_new.py -class Falls NotFalls -streams saliency spatial temporal -fid URFD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_SA.txt

python result_new.py -class Falls NotFalls -streams spatial temporal -fid URFD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB.txt

python result_new.py -class Falls NotFalls -streams pose temporal -fid URFD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_PO.txt
python result_new.py -class Falls NotFalls -streams pose spatial -fid URFD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RGB_PO.txt
python result_new.py -class Falls NotFalls -streams pose ritmo -fid URFD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RY_PO.txt
python result_new.py -class Falls NotFalls -streams pose saliency -fid URFD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_SA_PO.txt

python result_new.py -class Falls NotFalls -streams ritmo temporal -fid URFD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RY.txt
python result_new.py -class Falls NotFalls -streams ritmo spatial -fid URFD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RGB_RY.txt
python result_new.py -class Falls NotFalls -streams ritmo saliency -fid URFD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_SA_RY.txt

python result_new.py -class Falls NotFalls -streams saliency temporal -fid URFD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_SA.txt
python result_new.py -class Falls NotFalls -streams saliency spatial -fid URFD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RGB_SA.txt



python result_new.py -class Falls NotFalls -streams pose spatial temporal -fid URFD-test -cid URFD-train -f_classif svm_1 -fold 2 | tee ${dat}_test_${data}_SVM-1_OF_RGB_PO.txt
python result_new.py -class Falls NotFalls -streams ritmo spatial temporal -fid URFD-test -cid URFD-train -f_classif svm_1 -fold 2 | tee ${dat}_test_${data}_SVM-1_OF_RGB_RY.txt
python result_new.py -class Falls NotFalls -streams saliency spatial temporal -fid URFD-test -cid URFD-train -f_classif svm_1 -fold 2 | tee ${dat}_test_${data}_SVM-1_OF_RGB_SA.txt

python result_new.py -class Falls NotFalls -streams spatial temporal -fid URFD-test -cid URFD-train -f_classif svm_1 -fold 2 | tee ${dat}_test_${data}_SVM-1_OF_RGB.txt

python result_new.py -class Falls NotFalls -streams pose temporal -fid URFD-test -cid URFD-train -f_classif svm_1 -fold 2 | tee ${dat}_test_${data}_SVM-1_OF_PO.txt
python result_new.py -class Falls NotFalls -streams pose spatial -fid URFD-test -cid URFD-train -f_classif svm_1 -fold 2 | tee ${dat}_test_${data}_SVM-1_RGB_PO.txt
python result_new.py -class Falls NotFalls -streams pose ritmo -fid URFD-test -cid URFD-train -f_classif svm_1 -fold 2 | tee ${dat}_test_${data}_SVM-1_RY_PO.txt
python result_new.py -class Falls NotFalls -streams pose saliency -fid URFD-test -cid URFD-train -f_classif svm_1 -fold 2 | tee ${dat}_test_${data}_SVM-1_SA_PO.txt

python result_new.py -class Falls NotFalls -streams ritmo temporal -fid URFD-test -cid URFD-train -f_classif svm_1 -fold 2 | tee ${dat}_test_${data}_SVM-1_OF_RY.txt
python result_new.py -class Falls NotFalls -streams ritmo spatial -fid URFD-test -cid URFD-train -f_classif svm_1 -fold 2 | tee ${dat}_test_${data}_SVM-1_RGB_RY.txt
python result_new.py -class Falls NotFalls -streams ritmo saliency -fid URFD-test -cid URFD-train -f_classif svm_1 -fold 2 | tee ${dat}_test_${data}_SVM-1_SA_RY.txt

python result_new.py -class Falls NotFalls -streams saliency temporal -fid URFD-test -cid URFD-train -f_classif svm_1 -fold 2 | tee ${dat}_test_${data}_SVM-1_OF_SA.txt
python result_new.py -class Falls NotFalls -streams saliency spatial -fid URFD-test -cid URFD-train -f_classif svm_1 -fold 2 | tee ${dat}_test_${data}_SVM-1_RGB_SA.txt
