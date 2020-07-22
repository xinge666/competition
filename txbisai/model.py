def save_model(net,epoch,optimizer,model_name = 'best_transformer.pth'):
    if isinstance(net,torch.nn.DataParallel):
        net = net.module
    state = { 
                "epoch": epoch,
                "state": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_step":optimizer.step_num
            }   
    torch.save(state, model_name)
def test_model(model,test_loader,entroy):
    model = model.eval()
    predict_age = []
    predict_gender = []
    label_ages = []
    label_genders = []
    losses_age = []
    losses_gender= []
    with torch.no_grad():
        for i,(data) in enumerate(test_loader):

            user_id_feature,emb_feature,d_time,label_age,label_gender,user_id = data
            label_age = label_age.cuda()
            label_gender = label_gender.cuda()
            user_id_feature = user_id_feature.float().cuda()
            emb_feature = emb_feature.float().cuda()
            d_time = d_time.cuda()
            embs1,embs2 = model(user_id_feature,emb_feature,d_time)
            loss_age = entroy(embs1,label_age-1)
            loss_gender = entroy(embs2,label_gender-1)
            losses_age.append(loss_age.item())
            losses_gender.append(loss_gender.item())
            predict_age.extend(torch.max(embs1.cpu(),1)[1].numpy()+1)
            predict_gender.extend(torch.max(embs2.cpu(),1)[1].numpy()+1)
            label_ages.extend(label_age.cpu().numpy())
            label_genders.extend(label_gender.cpu().numpy())
            acc_age = accuracy_score(label_ages,np.floor(predict_age))
            acc_gender = accuracy_score(label_genders,np.floor(predict_gender))
    model.train()
    return acc_age,acc_gender,sum(losses_age)/len(losses_age),sum(losses_gender)/len(losses_gender)

