from Code.DTIDataset import DTIDataset

dataset = 'Davis'
file_path = 'data/' + dataset + '/processed'

fold = 1
epochs = 1000
batch = 4
lr = 1e-4

train_set = DTIDataset(dataset=dataset, compound_graph=file_path + '/train/fold/' + str(fold) + '/compound_graph.bin',
                       compound_id=file_path + '/train/fold/' + str(fold) + '/compound_id.npy',
                       protein_graph=file_path + '/train/fold/' + str(fold) + '/protein_graph.bin',
                       protein_embedding=file_path + '/train/fold/' + str(fold) + '/protein_embedding.npy',
                       protein_id=file_path + '/train/fold/' + str(fold) + '/protein_id.npy',
                       label=file_path + '/train/fold/' + str(fold) + '/label.npy')
x = train_set[0]
print(x)