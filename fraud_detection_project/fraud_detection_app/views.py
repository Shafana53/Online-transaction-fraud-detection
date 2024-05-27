from django.shortcuts import render,redirect
from sklearn.preprocessing import LabelEncoder
from django.http import JsonResponse
from .models import Transaction
from django.http import HttpResponse
import pandas as pd
import joblib
def home(request):
    return render(request,'front.html')
def front(request):
    return redirect(request, 'index.html')
def detect_fraud(request):
    
    if request.method == 'POST':
        # Get input from the form
        transaction_data = {
            
            
            'type': request.POST['type'],
            'amount': float(request.POST['amount']),
            'nameOrig': request.POST['nameOrig'],
            'oldbalanceOrg': float(request.POST['oldbalanceOrg']),
            'newbalanceOrig': float(request.POST['newbalanceOrig']),
            'nameDest': request.POST['nameDest'],
            'oldbalanceDest': float(request.POST['oldbalanceDest']),
            'newbalanceDest': float(request.POST['newbalanceDest']),
            
        }


        # Save the input data to the database
        new_transaction = Transaction.objects.create(**transaction_data)

        # Load the pre-trained  model
        model = joblib.load(r'fraud_model.pkl')

        # Apply label encoding to the 'type' feature
        label_encoder = LabelEncoder()
        transaction_data['type'] = label_encoder.fit_transform([transaction_data['type']])[0]

        # Drop unnecessary features not used during training
        features_to_drop = ['nameOrig', 'nameDest']
        input_data = pd.DataFrame([transaction_data]).drop(features_to_drop, axis=1)

        # Make a prediction
        is_fraud = model.predict(input_data)[0]     
          
       

        # Update the isFraud field in the database
        new_transaction.isFraud = is_fraud
        new_transaction.save()
        
        # Redirect to a new view that displays the result
        print("Debug: Redirecting to result view")
        return redirect('result',{'is_fraud': bool(is_fraud)})
        
    return JsonResponse({'error': 'Invalid request method'})

def index(request):
    return render(request, 'index.html')
    

def result(request, is_fraud):
    print("Debug: is_fraud =", is_fraud)
    return render(request, 'result.html', {'is_fraud': is_fraud})
