dat=$(date +"%F_%H:%M:%S")
data='FDD'

python result_new.py -class Falls NotFalls -streams pose -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_PO.txt

# python result_new.py -class Falls NotFalls -streams pose spatial temporal -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_PO.txt
# python result_new.py -class Falls NotFalls -streams ritmo spatial temporal -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_RY.txt
# python result_new.py -class Falls NotFalls -streams saliency spatial temporal -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_OF_RGB_SA.txt

# python result_new.py -class Falls NotFalls -streams spatial temporal -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_OF_RGB.txt

# python result_new.py -class Falls NotFalls -streams pose temporal -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_OF_PO.txt
# python result_new.py -class Falls NotFalls -streams pose spatial -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_RGB_PO.txt
# python result_new.py -class Falls NotFalls -streams pose ritmo -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_RY_PO.txt
# python result_new.py -class Falls NotFalls -streams pose saliency -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_SA_PO.txt

# python result_new.py -class Falls NotFalls -streams ritmo temporal -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_OF_RY.txt
# python result_new.py -class Falls NotFalls -streams ritmo spatial -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_RGB_RY.txt
# python result_new.py -class Falls NotFalls -streams ritmo saliency -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_SA_RY.txt

# python result_new.py -class Falls NotFalls -streams saliency temporal -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_OF_SA.txt
# python result_new.py -class Falls NotFalls -streams saliency spatial -fid FDD-test -cid FDD-train -f_classif svm_avg | tee ${dat}_test_${data}_SVM-AVG_RGB_SA.txt



# python result_new.py -class Falls NotFalls -streams pose -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_OF_RGB_PO.txt

python result_new.py -class Falls NotFalls -streams pose spatial temporal -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_OF_RGB_PO.txt
# python result_new.py -class Falls NotFalls -streams ritmo spatial temporal -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_OF_RGB_RY.txt
# python result_new.py -class Falls NotFalls -streams saliency spatial temporal -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_OF_RGB_SA.txt

# python result_new.py -class Falls NotFalls -streams spatial temporal -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_OF_RGB.txt

# python result_new.py -class Falls NotFalls -streams pose temporal -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_OF_PO.txt
# python result_new.py -class Falls NotFalls -streams pose spatial -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_RGB_PO.txt
# python result_new.py -class Falls NotFalls -streams pose ritmo -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_RY_PO.txt
# python result_new.py -class Falls NotFalls -streams pose saliency -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_SA_PO.txt

# python result_new.py -class Falls NotFalls -streams ritmo temporal -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_OF_RY.txt
# python result_new.py -class Falls NotFalls -streams ritmo spatial -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_RGB_RY.txt
# python result_new.py -class Falls NotFalls -streams ritmo saliency -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_SA_RY.txt

# python result_new.py -class Falls NotFalls -streams saliency temporal -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_OF_SA.txt
# python result_new.py -class Falls NotFalls -streams saliency spatial -fid FDD-test -cid FDD-train -f_classif svm_1 | tee ${dat}_test_${data}_SVM-1_RGB_SA.txt

# echo 'Test over FDD dataset is done' | mail -s 'FDD Test is over' guilherme.vieira.leite@gmail.com
