curl -L "https://app.roboflow.com/ds/lFX3VApTfs?key=BLkUIMEVEu" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
rm README.roboflow.txt
curl -L "https://app.roboflow.com/ds/iB9OqcauCA?key=XxckTCy6oV" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
rm README.roboflow.txt
curl -L "https://app.roboflow.com/ds/dFNBgTO6Jc?key=ZdsquCz5p1" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
rm README.roboflow.txt
curl -L "https://app.roboflow.com/ds/RFREW5X7Bt?key=haCSKhd2pt" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
rm README.roboflow.txt
curl -L "https://app.roboflow.com/ds/jh3k3gSHmr?key=ZVNLl98EUl" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
rm README.roboflow.txt
curl -L "https://app.roboflow.com/ds/LKzkq5oK04?key=U5KaF37ejY" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
rm README.roboflow.txt
curl -L "https://app.roboflow.com/ds/bHPSdF04GW?key=mfsnraSGyK" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
rm README.roboflow.txt
curl -L "https://app.roboflow.com/ds/7eRX9YhiM7?key=BfR6xtS92g" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
rm README.roboflow.txt
curl -L "https://app.roboflow.com/ds/7f9MSFn9lN?key=M354a6S1aQ" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
rm README.roboflow.txt
curl -L "https://app.roboflow.com/ds/pG5F1ov7hG?key=lT2rZUniHE" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
rm README.roboflow.txt
mv valid val
mkdir train/ImageSets
mkdir train/ImageSets/Main
mkdir val/ImageSets
mkdir val/ImageSets/Main

rm "./val/Annotations/test_00000306_jpg.rf.1d95880dbd04556e076162460b2ee6c4.xml"
rm "./val/JPEGImages/test_00000306_jpg.rf.1d95880dbd04556e076162460b2ee6c4.jpg"
rm "./train/Annotations/41_Swimming_Swimmer_41_174_jpg.rf.246bafe4be2de06c11d22db2842dede6.xml"
rm "./train/JPEGImages/41_Swimming_Swimmer_41_174_jpg.rf.246bafe4be2de06c11d22db2842dede6.jpg"
rm "./train/Annotations/test_00001352_jpg.rf.eb104eb711e7418060bec66fbf3c38dc.xml"
rm "./train/JPEGImages/test_00001352_jpg.rf.eb104eb711e7418060bec66fbf3c38dc.jpg"

python reorganize_dataset.py
