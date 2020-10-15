import os
import shutil
from preprocess_small_datasets import preprocess_all_small_datasets


def prepare_all_small_datasets(args):

    preprocess_all_small_datasets(args)

    for dataset in ['3dpes', 'cuhk01', 'cuhk02', 'ilids', 'prid', 'viper']:
        
        os.mkdir('data/'+dataset+'/pytorch')

        files = ['data/'+dataset+'/train.txt', 'data/'+dataset+'/val.txt']
        with open('data/'+dataset+'/train_all.txt', 'w') as outfile:
            for fname in files:
                with open(fname) as infile:
                    outfile.write(infile.read())

        for file in os.listdir('data/'+dataset):
            if file.split('.')[-1]=='txt':
                name = file.split('.')[0]

                if 'gallery' in name:
                    name = 'gallery'
                if 'probe' in name: 
                    name = 'query'


                data_dir = 'data/'+dataset+'/pytorch/'+name
                os.mkdir(data_dir)

                with open('data/'+dataset+'/'+file) as f:
                    for line in f:
                        img,label = line.split()

                        if not os.path.exists(data_dir+'/'+label):
                            os.mkdir(data_dir+'/'+label)
                        a,b,c,d = img.split('/')
                        shutil.copyfile(img, data_dir+'/'+label+'/'+c+'_'+d)







