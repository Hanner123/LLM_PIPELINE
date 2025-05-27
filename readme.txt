vorher: tiny_train mit hugging face modulen genutzt, der ganze code (measure, export, onnx_quant gehört dazu), quant16 hat funktioniert (nur tensorrt war ausreichend, parameter mit 16bit angegeben)
quant8 hat nicht funktioniert, das modell musste vorher mit onnx_quant quantisiert werdern, das hat bis zum export funktioniert, aber das exportierte Modell konnte ich nicht mehr mir Tensorrt öffnen ("nicht symmetrisch quantisiert"), 
obwohl ich den Parameter auf symmetrisch gestetzt habe und dann auch versucht habe mit graphsurgeon das Modell irgendwie anzupassen

Ich brauche jetzt ein eigenes Modell (keine hugging face module), damit ich die für brevitas vorverarbeiten kann (quantlinear). Dafür verwende ich jetzt das "große" bert modell, weil ich davon code und die einzelnen Klassen habe.
Was macht dieses BERT Modell eigentlich? Lückentext raten???
Während des Trainings maskiert das Modell zufällig bestimmte Wörter in einem Satz und versucht dann, diese zu erraten. Dies verbessert das Verständnis des Modells für den Gesamtzusammenhang eines Textes erheblich.
---> am besten teste ich es dann auch damit


Erster Schritt: Training von Bert ans laufen bekommen
- das training "funktioniert", aber es läuf sehr lange (pro Epoche 20min) und die accuracy verschlechtert sich eher

Zweiter Schritt: Training von Bert Quant - funktioniert

Exportieren als onnx (hat ewig gedauert, korrekte speicherung vom Tokenizer und pfaden allgemein) und in netron anschauen -> sieht groß aus...



Dritter schritt: quantisierung mit brevitas (erstmal vorziehen) - hat funktioniert

Vierter schritt: Evaluierung darauf auslegen (inputs, accuracy..)
 - scheinbar soll das Bert modell fragen und antworten voneinander unterscheiden können. Ich verstehe aber nicht so ganz wo die labels gespeichert sein sollen und wie ich das teste...
 - oder maskierte Wörter ersetzen - wie teste ich das?? 

Fünfter Schritt: funktioniert die Evaluierung mit Tensorrt immer noch? da gibts ganz spezielle Regeln..
 - anderer Datensatz zum testen (tokenized dataset irgendwie vorher speichern)
 - ansonsten ähnlich, nur mit 8 bit
 - auch noch das nicht - quantisierte modell testen


