# bare minimum to illustrate how to extract features using the trained model. 

1. call the main function in feat_extractor - returns 1024 or 527 dimensional features

2. It will work with audio of any duration but I would suggest to keep it more than 1.5 seconds for now. 


3. You can turn on gpu use by 'usegpu' variable. Although, for very long audio (more than a few minutes) you might end up getting gpu memory error. 


4. class names and id in classes_id_name.txt for the 527 sounds in audioset over which the model was trained. 

6. Doubts??? -  send me an email. 
