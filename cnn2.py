from torchvision import transforms
import torch
from model import EncoderCNN, DecoderRNN
from vocabulary import Vocabulary

def clean_sentence(vocab, output):
    list_string = []
    
    for idx in output:
        list_string.append(vocab.idx2word[idx])
    
    list_string = list_string[1:-1] # Discard <start> and <end> words
    sentence = ' '.join(list_string) # Convert list of string to full string
    sentence = sentence.capitalize()  # Capitalize the first letter of the first word
    return sentence

def Captioning(path):
    # TODO #1: Define a transform to pre-process the testing images.
    transform_test = transforms.Compose([transforms.Resize((224, 224)), \
                                        transforms.ToTensor(), \
                                        transforms.Normalize((0.485, 0.456, 0.406), \
                                                            (0.229, 0.224, 0.225))])

    # TODO #2: Specify the saved models to load.
    encoder_file = 'encoder-step18400.pkl' 
    decoder_file = 'decoder-step18400.pkl'

    # TODO #3: Select appropriate values for the Python variables below.
    embed_size = 512
    hidden_size = 512

    #cocoapi_loc='C:\\Users\\trong\\Downloads\\annotations_trainval2017'

    annotations_file = 'captions_train2017.json'#cocoapi_loc + '/annotations/captions_train2017.json'
    # The size of the vocabulary.
    vocab = Vocabulary(6, "./vocab.pkl", "<start>",
                "<end>", "<unk>", annotations_file, True)
    vocab_size = len(vocab)

    # Initialize the encoder and decoder, and set each to inference mode.
    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    # Load the trained weights.
    encoder.load_state_dict(torch.load(encoder_file, map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(decoder_file, map_location=torch.device('cpu')))

    from PIL import Image
    img = Image.open(path)
    #image = image.to(device)
    img = transform_test(img).unsqueeze(0)
    # Obtain the embedded image features.
    print("image.shape: ", img.shape)
    features = encoder(img).unsqueeze(1)
    print("features.shape: ", features.shape)
    print()

    # Pass the embedded image features through the model to get a predicted caption.
    output = decoder.sample(features)
    print('example output:', output)

    assert (type(output)==list), "Output needs to be a Python list" 
    assert all([type(x)==int for x in output]), "Output should be a list of integers." 
    assert all([x in vocab.idx2word for x in output]), "Each entry in the output needs to correspond to an integer that indicates a token in the vocabulary."

    sentence = clean_sentence(vocab, output)
    return sentence