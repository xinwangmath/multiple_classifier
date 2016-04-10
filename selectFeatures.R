# Author: Xin Wang
# Email: xinwangmath@gmail.com

selectFeatures = function(candidate, target, del = "_"){
	#survive_index = 1:length(target)
	target_index = 1:length(target)


	cand = sapply(candidate, function(xx){
		return(paste(unlist(strsplit(xx, del)), sep = "", collapse = ""))
		})

	targ = sapply(target, function(xx){
		return(paste(unlist(strsplit(xx, "_")), sep = "", collapse = ""))
		})

	r_targ = sapply(targ, function(xx){
		return(paste( c("r", xx), sep ="", collapse = ""))
		})
	l_targ = sapply(targ, function(xx){
		return(paste( c("l", xx), sep ="", collapse = ""))
		})

	survive_index = sapply(target_index, function(xx){
		
		if(any(cand == targ[xx])){
			return(which(cand == targ[xx]))
		} else if( any(cand == r_targ[xx]) ){
			return(which(cand == r_targ[xx]))
		} else{
			return(which(cand == l_targ[xx]))
		}

		})

	return(candidate[survive_index])



}