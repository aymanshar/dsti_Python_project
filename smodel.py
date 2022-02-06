'''
    help on smodel 
    --------------
    package content
    ---------------
    - text_preprocessing(s)      
    - bert_model(pd, colunmName_text, colunmName_category, second_category,third_category,furth_category )
    - bert_predict(test_col)
    - predict_con_multiline(txt)
    - regresion(pd=None,train_word=None,train_label=None,test_word=None,test_label=None)
    - clean_text(text)
    
'''

def text_preprocessing(s):
    import re
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove trailing whitespace
    """
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"n\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    #s = " ".join([word for word in s.split()
    #              if word not in stopwords.words('english')
    #              or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s

# main function for bert inference 
def bert_model(pd, colunmName_text, colunmName_category, second_category,third_category,fourth_category):
    '''
    Bert inference 
        >> to load the Pre-trained BERT the light version DistilBERT model
            
        used arguments: 
            - pd : panda instance from import pandas as pd
            - ColunmName_text: the column will be used for training df['col'] or df[''][:] if part of data
            - ColunmName_category: category column df['category'] or df['category'][:]
            - second_category : second category for classification coloumn.
            - third_category : second category for classification coloumn.
            - fourth_category : second category for classification coloumn.

    '''
    
    import transformers as ppb
    
    global tokenizer
    global model
    
    # For DistilBERT:
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    # Load pretrained distilBERT model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    print('\nTransformation: creating model instance ... steps\n')
    model = model_class.from_pretrained(pretrained_weights)
    print('\n-------------------Transformation Done -------------------')
    #model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    #create new column in dataframe for Tokenization This turns every sentence into the list of ids. saving in new column name tokenized
    print (f"Transform column to tokens words then numbers", '\n----------------------------------')
    tokenized = colunmName_text.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    print('\nby tokenized first 3 records like this : \n\n' , tokenized[0:3])
    print('\n===============================================================')

    # berting(datafram['kenized'],data_size,1)
    import numpy
    global padded
    global attention_mask
                    
    max_len = 0
    for i in tokenized:
        if len(i) > max_len:
            max_len = len(i)

    print('\n padding ... step','\n----------------------')
    padded= numpy.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    print('\npadded shape: ', numpy.array(padded).shape)
    
    print('\n masking ... step','\n----------------------')
    attention_mask = numpy.where(padded != 0, 1, 0)
    print('\nattention_mask shape: ',attention_mask.shape)

    print('\ntensoring ... step','\n----------------------')

    import torch
    # Processing with DistilBERT
    # We now create an input tensor out of the padded token matrix, and send that to DistilBERT
    global features
    global labels 
    
    # Convert padded array to tensors
    input_ids = torch.tensor(padded)  
    print('\ninput_ids \n','----------------------\n',input_ids)
    
    attention_mask_tens = torch.tensor(attention_mask)
    print('\nattention_mask \n','----------------------\n',attention_mask_tens)
    
    print('\ntorch.no_grad ... step','\n-------------------------')
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask_tens)
    
    #torch.save(model, 'path/to/model')
    print('\nlast_hidden \n','----------------------\n',last_hidden_states[0][:,0,:])

    print('\npreparing features and labels ... step','\n-------------------------')
    features = last_hidden_states[0][:,0,:].numpy()
    
    # save numpy array as csv file
    from numpy import asarray
    from numpy import savetxt
    # save to csv file
    savetxt('/content/drive/MyDrive/sereniiti_project/model_save/Seneriiti_featuers.csv', features, delimiter=',')
    savetxt('/content/drive/MyDrive/sereniiti_project/model_save/Seneriiti_padded.csv', padded, delimiter=',')
    savetxt('/content/drive/MyDrive/sereniiti_project/model_save/Seneriiti_masks.csv', attention_mask, delimiter=',')
    print ('\nfeatuers, padded and attention masks saved successfully\n')
    labels = torch.tensor(colunmName_category)
    labels2 = torch.tensor(second_category)
    labels3 = torch.tensor(third_category)
    labels4 = torch.tensor(fourth_category)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', colunmName_text[0])
    print('Token IDs:', input_ids[0])
    
    # Convert our train and validation features to InputFeatures that BERT understands.
    from sklearn.model_selection import train_test_split
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
    train_features2, test_features2, train_labels2, test_labels2 = train_test_split(features, labels2)
    train_features3, test_features3, train_labels3, test_labels3 = train_test_split(features, labels3)
    train_features4, test_features4, train_labels4, test_labels4 = train_test_split(features, labels4)
    # logistic Regression model    
    from sklearn.linear_model import LogisticRegression
    model_lr = LogisticRegression()
    model_lr2 = LogisticRegression()
    model_lr3 = LogisticRegression()
    model_lr4 = LogisticRegression()
    
    # model fit train feature and train labels
    model_lr.fit(train_features, train_labels)
    model_lr2.fit(train_features2, train_labels2)
    model_lr3.fit(train_features3, train_labels3)
    model_lr4.fit(train_features4, train_labels4)
    
    print('\nlen of train_features ratings: ',len(train_features))
    print('\nlen of train_features steps: ',len(train_features2))
    print('\nlen of train_features tips: ',len(train_features3))
    print('\nlen of train_features tips_group: ',len(train_features4))
    print('\nlogisticRegression for the training set ... step','\n==============================')
    
    # predict
    print('\npredict the test features ... step','\n============================')
    predict_md=model_lr.predict(test_features)
    predict_md2=model_lr2.predict(test_features2)
    predict_md3=model_lr3.predict(test_features3)
    predict_md4=model_lr4.predict(test_features4)

    from sklearn.metrics import accuracy_score #
    # check the accuracy of the test labels with the predicted 
    score=accuracy_score(predict_md,test_labels)
    score2=accuracy_score(predict_md2,test_labels2)
    score3=accuracy_score(predict_md3,test_labels3)
    score4=accuracy_score(predict_md4,test_labels4)
    print('\naccuracy score of rating: ', score,'\naccuracy score of steps: ', score2,'\naccuracy score of tips: ', score3,'\naccuracy score of tips group: ', score4)

    print('\n Confusion_matrix rating : \n', '--------------------\n\n',pd.crosstab(test_labels, predict_md, rownames=['True'], colnames=['Predicted'], margins=True))
    print('\n Confusion_matrix steps : \n', '--------------------\n\n',pd.crosstab(test_labels2, predict_md2, rownames=['True'], colnames=['Predicted'], margins=True))
    print('\n Confusion_matrix tips : \n', '--------------------\n\n',pd.crosstab(test_labels3, predict_md3, rownames=['True'], colnames=['Predicted'], margins=True))
    print('\n Confusion_matrix tips_groups : \n', '--------------------\n\n',pd.crosstab(test_labels4, predict_md4, rownames=['True'], colnames=['Predicted'], margins=True))
    print('\n====================================')
    #from sklearn.metrics import classification_report
    #matrix=classification_report (predict_md, test_labels)
    print('\nSaving tokinizer: \n','====================\n\n')
    import os

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

    output_dir = '/content/drive/MyDrive/sereniiti_project/model_save/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    print('\n\n------ Done-----\n')

# for bert prediction and testing
def bert_predict(test_col):
    '''
    Bert predicting
    
        >> to predict category, step, tips, tips_groups for sentences
            
        arguments: 
            - test_col : the text to predict for

    '''
    
    import transformers as ppb
    
    global tokenizer
    global model

    import os
    output_dir = '/content/drive/MyDrive/sereniiti_project/model_save/'
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(output_dir)
    tokenizer = tokenizer_class.from_pretrained(output_dir)
                                        
    #create new column in dataframe for Tokenization This turns every sentence into the list of ids. saving in new column name tokenized
    #tokenized = colunmName_text.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    tokenized_test = test_col.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    import numpy
    max_len=85
    
    from numpy import loadtxt
    #padded= numpy.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    padded= loadtxt('/content/drive/MyDrive/sereniiti_project/model_save/Seneriiti_padded.csv', delimiter=',')
    padded_test= numpy.array([i + [0]*(max_len-len(i)) for i in tokenized_test.values])
    
    #attention_mask = numpy.where(padded != 0, 1, 0)
    attention_mask = loadtxt('/content/drive/MyDrive/sereniiti_project/model_save/Seneriiti_masks.csv', delimiter=',')
    attention_mask_test = numpy.where(padded_test != 0, 1, 0)
    
    import torch
    input_ids = torch.tensor(padded) 
    input_ids_test = torch.tensor(padded_test)  
    
    attention_mask_tens = torch.tensor(attention_mask)
    attention_mask_tens_test = torch.tensor(attention_mask_test)

    # load saved last hidden from csv file
    features_load = loadtxt('/content/drive/MyDrive/sereniiti_project/model_save/Seneriiti_featuers.csv', delimiter=',')
    print('\n feature loaded Succesfully \n',len(features_load))

    with torch.no_grad():
        last_hidden_states_test = model(input_ids_test, attention_mask=attention_mask_tens_test)

    import pandas as db
    import os
    input_file = '/content/drive/MyDrive/sereniiti_project/model_save/Seneriiti_dataset_l.csv'
    data=db.read_csv(input_file)
    features = features_load
    features_test = last_hidden_states_test[0][:,0,:].numpy()
    labels = data['rating'][:len(features_load)]
    labels2 = data['step_code'][:len(features_load)]
    labels3 = data['tips_code'][:len(features_load)]
    labels4 = data['tips_groups_code'][:len(features_load)]
    
    from sklearn.model_selection import train_test_split
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
    train_features2, test_features2, train_labels2, test_labels2 = train_test_split(features, labels2)
    train_features3, test_features3, train_labels3, test_labels3 = train_test_split(features, labels3)
    train_features4, test_features4, train_labels4, test_labels4 = train_test_split(features, labels4)
    
    # logistic Regression model    
    from sklearn.linear_model import LogisticRegression
    model_lr = LogisticRegression()
    model_lr2 = LogisticRegression()
    model_lr3 = LogisticRegression()
    model_lr4 = LogisticRegression()
    
    # model fit train feature and train labels
    model_lr.fit(train_features, train_labels)
    model_lr2.fit(train_features2, train_labels2)
    model_lr3.fit(train_features3, train_labels3)
    model_lr4.fit(train_features4, train_labels4)
    
    global predict_md_test
    global predict_md_test2
    global predict_md_test3
    global predict_md_test4
    
    # predict
    predict_md_test = model_lr.predict(features_test)
    predict_md_test2 = model_lr2.predict(features_test)
    predict_md_test3 = model_lr3.predict(features_test)
    predict_md_test4 = model_lr4.predict(features_test)
    
    return predict_md_test,predict_md_test2,predict_md_test3,predict_md_test4

def predict_con_multiline(txt):
  import pandas as db
  data=db.read_csv('/content/drive/MyDrive/sereniiti_project/model_save/Seneriiti_dataset_l.csv')
  x_new_df=db.DataFrame({'token':txt.split(",")})
  
  print ('\n****** It will take some time ******\n','...... Please wait ......'  )
  # call predict function
  bert_predict(x_new_df['token'] )
  x_new_df['predicting_ratings']= predict_md_test
  x_new_df['rating']=x_new_df['predicting_ratings'].map({0:'very likely to elicit a defensive reaction or high pain in other people', 1:'likely to elicit a defensive reaction in other people',2:'slight tendance to elicit defensiveness in other people', 3:'slight tendance to encourage dialogue',4:'likely to encourage an open dialogue or neutral',5:'very likely to encourage an open dialogue which is mutually satisfying'})
  
  # add the predicted values as steps to the dataframe
  x_new_df['predicting_steps']= predict_md_test2
  # add the predicted values as tips to the dataframe
  x_new_df['predicting_tips']= predict_md_test3
  # add the predicted values as tips group to the dataframe
  x_new_df['predicting_tips_groups']= predict_md_test4

  # map the step code values with step 
  step_code= {}
  # map the stips code values with tips 
  tips_code= {}
  # map the step code values with step 
  tips_groups_code= {}
  for i in range(len(data)):
    s={data.loc[i, "step_code"] : data.loc[i, "steps"]}
    step_code.update(s)
    v={data.loc[i, "tips_code"] : data.loc[i, "tips"]}
    tips_code.update(v)
    u={data.loc[i, "tips_groups_code"] : data.loc[i, "tips_groups"]}
    tips_groups_code.update(u)

  x_new_df['steps_classification']=x_new_df['predicting_steps'].map(step_code)  
  x_new_df['tips_classification']=x_new_df['predicting_tips'].map(tips_code) 
  x_new_df['tips_groups_classification']=x_new_df['predicting_tips_groups'].map(tips_groups_code)
  
  # create new colum for result combining all results
  x_new_df['result']='\n'+x_new_df['token']+'\n============================='+'\n** Rating as: '+ x_new_df['rating'] +' \n** steps Classified as: '+ x_new_df['steps_classification'] +' \n** Tips classified as: '+ x_new_df['tips_classification']+' \n** Tips Groups classified as: '+ x_new_df['tips_groups_classification']+'\n'

  print ('\nResult abelow: \n', '====================\n')
  for item in x_new_df['result']:
    print(item)

# regression  
def regresion(pd=None,train_word=None,train_label=None,test_word=None,test_label=None):
    '''
    regresion(pd,train_word,train_label,test_word,test_label)
    ---------------------------------------------------------
        >> logistic regression model to train and test data set
            
        arguments:
            - pd : passing panda instance to be used inside the function
            - train_word
            - train_label
            - test_word
            - test_label
        variabels :
            - model_lr
                model_lr = LogisticRegression()
                model_lr.fit(train_word, train_label)
            - predict_md
                predict_md=model_lr.predict(test_word)

    rturned results classification_report, Confusion_matrix, accuracy score
    '''
    if (pd is None or train_word is None or train_label is None or test_word is None or test_label is None):
        print("one or all arguments are missing, you must pass all needed arguments ")
    else:
        # logistic Regression model
    
        global model_lr
        global predict_md
        
        from sklearn.linear_model import LogisticRegression

        model_lr = LogisticRegression()
        # model fit train feature and train labels
        model_lr.fit(train_word, train_label)

        # predict
        predict_md=model_lr.predict(test_word)

        
        from sklearn.metrics import accuracy_score #
        # check the accuracy of the test labels with the predicted 
        score=accuracy_score(predict_md,test_label)
        print('accuracy score:', score)

        print('\n Confusion_matrix : \n\n', pd.crosstab(test_label, predict_md, rownames=['True'], colnames=['Predicted'], margins=True))

        from sklearn.metrics import classification_report
        matrix=classification_report (predict_md, test_label)
        print('\nclassification_report: \n\n',matrix)

def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """
    import re
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # remove the characters [\], ['] and ["] replacing by space ""
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()
    
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text