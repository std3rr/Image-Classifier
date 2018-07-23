from train import Network
import argparse
import sys
import json

def main():
    
    in_args = get_args()

    netw = Network(gpu=in_args.gpu, checkpoint=in_args.checkpoint)
    print(f'predicting {in_args.image} top{in_args.topk}') 
    
    # If we're geting cat idx from path, print out ground truth
    try:
        cat_nr = in_args.image.split('/')[-2]
        if cat_nr.isdigit():
            cat = in_args.cat_to_name[cat_nr]
            print(f"Ground thruth category: {cat}")
    except:
        None
        
    probs, classes, _ = netw.predict(in_args.image, topk=in_args.topk)
    labels = [in_args.cat_to_name[i] for i in classes]
    print('\n')
    for name, pred in zip(labels, probs):
        print("{:16s}\t: {:3.5f}".format(name, pred*100))
    
def get_args():
    """
    Get command line arguments and return a dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('image',help='Image to predict')
    parser.add_argument('checkpoint',help='checkpoint model to use')
    parser.add_argument('--topk',type=int,default=5,help='Top number predictions to display')
    parser.add_argument('--gpu',type=bool, const=True, nargs='?',default=False, help='Enable gpu/cuda mode')
    parser.add_argument('--cat_to_name',type=str, default='cat_to_name.json', help='Use custom json file for categories')
        
    parsed_args = parser.parse_args()
    
    with open(parsed_args.cat_to_name, 'r') as f:
        parsed_args.cat_to_name = json.load(f)    
    
    if not parsed_args.checkpoint:
        print("Need a model checkpoint to load.")
        print("Available checkpoints:",
              "----------------------")
        print(sys.listdir("checkpoint/"))
        exit()
    
    return parsed_args

if __name__ == "__main__":
    main()


