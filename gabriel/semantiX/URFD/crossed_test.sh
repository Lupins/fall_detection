dat=$(date +"%F_%H:%M:%S")
data='URFD->FDD'

#python result_new.py -class Falls NotFalls -streams pose spatial temporal -fid FDD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_PO.txt
#python result_new.py -class Falls NotFalls -streams ritmo spatial temporal -fid FDD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_RY.txt
#python result_new.py -class Falls NotFalls -streams saliency spatial temporal -fid FDD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_SA.txt

#python result_new.py -class Falls NotFalls -streams spatial temporal -fid FDD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB.txt

#python result_new.py -class Falls NotFalls -streams pose temporal -fid FDD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_PO.txt
#python result_new.py -class Falls NotFalls -streams pose spatial -fid FDD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RGB_PO.txt
#python result_new.py -class Falls NotFalls -streams pose ritmo -fid FDD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RY_PO.txt
#python result_new.py -class Falls NotFalls -streams pose saliency -fid FDD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_SA_PO.txt

#python result_new.py -class Falls NotFalls -streams ritmo temporal -fid FDD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RY.txt
#python result_new.py -class Falls NotFalls -streams ritmo spatial -fid FDD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RGB_RY.txt
#python result_new.py -class Falls NotFalls -streams ritmo saliency -fid FDD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_SA_RY.txt

#python result_new.py -class Falls NotFalls -streams saliency temporal -fid FDD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_SA.txt
#python result_new.py -class Falls NotFalls -streams saliency spatial -fid FDD-test -cid URFD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RGB_SA.txt


data='FDD->URFD'
python result_new.py -class Falls NotFalls -streams pose spatial temporal -fid URFD-test -cid FDD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_PO.txt
python result_new.py -class Falls NotFalls -streams ritmo spatial temporal -fid URFD-test -cid FDD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_RY.txt
python result_new.py -class Falls NotFalls -streams saliency spatial temporal -fid URFD-test -cid FDD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_SA.txt

python result_new.py -class Falls NotFalls -streams spatial temporal -fid URFD-test -cid FDD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RGB.txt

python result_new.py -class Falls NotFalls -streams pose temporal -fid URFD-test -cid FDD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_PO.txt
python result_new.py -class Falls NotFalls -streams pose spatial -fid URFD-test -cid FDD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RGB_PO.txt
python result_new.py -class Falls NotFalls -streams pose ritmo -fid URFD-test -cid FDD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RY_PO.txt
python result_new.py -class Falls NotFalls -streams pose saliency -fid URFD-test -cid FDD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_SA_PO.txt

python result_new.py -class Falls NotFalls -streams ritmo temporal -fid URFD-test -cid FDD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_RY.txt
python result_new.py -class Falls NotFalls -streams ritmo spatial -fid URFD-test -cid FDD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RGB_RY.txt
python result_new.py -class Falls NotFalls -streams ritmo saliency -fid URFD-test -cid FDD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_SA_RY.txt

python result_new.py -class Falls NotFalls -streams saliency temporal -fid URFD-test -cid FDD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_OF_SA.txt
python result_new.py -class Falls NotFalls -streams saliency spatial -fid URFD-test -cid FDD-train -f_classif svm_avg -fold 2 | tee ${dat}_test_${data}_SVM-AVG_RGB_SA.txt
