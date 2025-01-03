# Migliorare-un-NPC-in-un-gioco-a-turni-tramite-LLM-e-RL
I Non-Player Character (NPC), o personaggi non giocanti, sono entità presenti nei videogiochi, gestite dal sistema, che contribuiscono a rendere il mondo di gioco più immersivo e realistico, arricchendo l’esperienza complessiva del giocatore. Ad esempio, in un videogioco che presenta combattimenti a turni, la presenza di un NPC con il quale conversare per ottenere suggerimenti è un qualcosa che potrebbe rendere maggiormente interattivo lo scontro. Gli NPC sono progettati per interagire con i giocatori in molteplici modi, ricoprendo ruoli diversi. Tuttavia, sebbene siano una componente essenziale nei videogiochi, la loro implementazione può spesso risultare limitata, con comportamenti statici e dialoghi poco realistici. Questo accade perché si basano su set predefiniti di risposte che, durante lunghe sessioni di gioco, possono diventare ripetitivi e frustranti per il giocatore.
Questo progetto di tesi affronta la possibilità di usare i Large Language Models (LLM), in particolare FlanT5-Large, ed il Reinforcement Learning (RL), in particolare il Proximal Policy Optimization, per creare un NPC in grado di assistere un giocatore, rappresentato da un agente, suggerendo la prossima azione da intraprendere. Uno dei principali limiti degli LLM è la tendenza a generare allucinazioni, ovvero risposte sintatticamente corrette ma disconnesse dalla realtà. Per alleviare il problema delle allucinazioni, è stato chiesto al NPC di fornire il suo reasoning e, grazie ad un ulteriore agente di LLM addestrato tramite Proximal Policy Optimization (PPO), sono stati forniti istruzioni per effettuare il rephrasing della risposta iniziale, con l’obiettivo di rendere i consigli più efficaci possibili. Per testare l’efficacia dell’approccio proposto sono stati condotti esperimenti su un numero variabile di episodi. Inoltre, è stato condotto un test con una configurazione in due fasi in cui, nella prima, l’agente opera in autonomia per un determinato numero di episodi, mentre nella seconda è supportato dalla metodologia proposta, al fine di analizzare l’impatto dei consigli sulla strategia finale. I risultati dimostrano un significativo incremento nella qualità delle azioni proposte, evidenziando il potenziale dell’integrazione tra LLM e RL per migliorare l’interazione con un NPC nei videogiochi.
