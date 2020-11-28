from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd 
# Create your views here.
from sklearn.externals import joblib
reloadModel = joblib.load('\Model\nlp1.py')
def index(request):
    context = {'a':'Sentimental'}
    return render(request,'index.html',context)
    #return HttpResponse({'a':1})


def predictMPG(request):
    if request.method == 'POST':
        print(request.POST.dict())
        request.POST.get('TEXT')
matrix = confusion_matrix( y_test.argmax(axis=1), predictions.argmax(axis=1))
sentiment = classification_report(y_test ,predictions)
accuracy = accuracy_score(y_test, predictions)
context = {'matrix':'matrix'}
context = {'sentiment':'sentiment'}
context = {'accuracy':'accuracy'}
return render(request,'index.html',context)

